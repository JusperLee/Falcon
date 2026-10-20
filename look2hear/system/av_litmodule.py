
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from .. import losses


def flatten_dict(d, parent_key="", sep="_"):
    
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioVisualLightningModule(pl.LightningModule):
    def __init__(
        self,
        audio_model=None,
        video_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val_loss/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        # per-bucket val outputs: key = min(n_near, 4)
        self._val_bucket_outputs = {k: [] for k in range(5)}

    def forward(self, wav, mouth=None):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        if mouth.ndim == 4:
            # Grayscale, no color channel needed
            mouth = mouth.unsqueeze(1)
        with torch.no_grad():
            mouth_emb = self.video_model(mouth.type_as(wav))
        return self.audio_model(wav, mouth_emb)

    def training_step(self, batch, batch_nb):
        mixtures, targets, mouth, *_ = batch

        est_sources = self(mixtures, mouth)
        if targets.ndim == 2:
            targets = targets.unsqueeze(1)

        loss = self.loss_func["train"](est_sources, targets)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_nb, dataloader_idx):
        # cal val loss
        if dataloader_idx == 0:
            mixtures, targets, mouth, _, n_near_batch = batch
            est_sources = self(mixtures, mouth)
            if targets.ndim == 2:
                targets = targets.unsqueeze(1)

            # Overall batch loss (for val_loss / early stop)
            loss = self.loss_func["val"](est_sources, targets)
            self.log(
                "val_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.validation_step_outputs.append(loss)

            # Per-sample loss, bucketed by n_near
            B = mixtures.size(0)
            for i in range(B):
                # Single-sample loss: [1, n_src, T] vs [1, n_src, T]
                sample_loss = self.loss_func["val"](
                    est_sources[i:i+1], targets[i:i+1]
                )
                # Parse n_near for this sample
                if isinstance(n_near_batch, torch.Tensor):
                    n_near_i = int(n_near_batch[i].item())
                else:
                    # fallback: list or scalar
                    n_near_i = int(n_near_batch[i]) if hasattr(n_near_batch, '__len__') else int(n_near_batch)
                bucket = min(n_near_i, 4)
                self._val_bucket_outputs[bucket].append(sample_loss.detach())

            return {"val_loss": loss}

        # cal test loss
        if (self.trainer.current_epoch) % 10 == 0 and dataloader_idx == 1:
            mixtures, targets, mouth, *_ = batch
            est_sources = self(mixtures, mouth)
            if targets.ndim == 2:
                targets = targets.unsqueeze(1)
            tloss = self.loss_func["val"](est_sources, targets)
            self.log(
                "test_loss",
                tloss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.test_step_outputs.append(tloss)
            return {"test_loss": tloss}

    def on_validation_epoch_end(self):
        # val
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.logger.experiment.log(
            {"learning_rate": self.optimizer.param_groups[0]["lr"], "epoch": self.current_epoch}
        )
        self.logger.experiment.log(
            {"val_pit_sisnr": -val_loss, "epoch": self.current_epoch}
        )

        # per-bucket val_pit_sisnr (K=0,1,2,3,4+)
        _bucket_names = {0: "val_pit_sisnr_0near", 1: "val_pit_sisnr_1near",
                         2: "val_pit_sisnr_2near", 3: "val_pit_sisnr_3near",
                         4: "val_pit_sisnr_4plus_near"}
        for bucket, name in _bucket_names.items():
            outputs = self._val_bucket_outputs[bucket]
            if outputs:
                avg = torch.mean(torch.stack(outputs))
                avg = torch.mean(self.all_gather(avg))
                self.logger.experiment.log({name: -avg.item(), "epoch": self.current_epoch})
        for bucket in self._val_bucket_outputs:
            self._val_bucket_outputs[bucket].clear()

        # test
        if (self.trainer.current_epoch) % 10 == 0:
            avg_loss = torch.stack(self.test_step_outputs).mean()
            test_loss = torch.mean(self.all_gather(avg_loss))
            self.logger.experiment.log(
                {"test_pit_sisnr": -test_loss, "epoch": self.current_epoch}
            )
        self.validation_step_outputs.clear()  # free memory
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return [self.val_loader, self.test_loader]

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic