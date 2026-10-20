import os
import sys
import torch
torch.cuda.empty_cache()

import argparse
import json
import yaml
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau

import look2hear.models
import look2hear.losses
import look2hear.system
import look2hear.videomodels
import look2hear.utils
from look2hear.system import make_optimizer
from look2hear.utils import print_only

from look2hear.datas.avspeech_dataset_near import AVSpeechDataModuleWithNear
from look2hear.system.av_litmodule_nass_v2 import AudioVisualLightningModuleNASSv2

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="configs/NASSv2_lrs3_complex_nonmoving.yml",
    help="Path to YAML config file",
)
parser.add_argument(
    "--ckpt_path",
    default="",
    help="Optional checkpoint path to resume from (leave empty to train from scratch)",
)


def main(config, ckpt_path=""):
    use_near = config["datamodule"]["data_config"].get("use_near", False)
    print_only(f"[audio_train_nass_v2] use_near = {use_near}")

    # ── DataModule ────────────────────────────────────────────────────────────
    print_only(f"Instantiating datamodule: AVSpeechDataModuleWithNear (use_near={use_near})")
    datamodule = AVSpeechDataModuleWithNear(**config["datamodule"]["data_config"])
    datamodule.setup()
    train_loader, val_loader, test_loader = datamodule.make_loader

    # ── Audio model ───────────────────────────────────────────────────────────
    print_only(f"Instantiating AudioNet <{config['audionet']['audionet_name']}>")
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )

    # ── Video model ───────────────────────────────────────────────────────────
    video_model = getattr(look2hear.videomodels, config["videonet"]["videonet_name"])(
        **config["videonet"]["videonet_config"],
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    print_only(f"Instantiating Optimizer <{config['optimizer']['optim_name']}>")
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(f"Instantiating Scheduler <{config['scheduler']['sche_name']}>")
        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
            optimizer=optimizer, **config["scheduler"]["sche_config"]
        )

    # ── Experiment dir ────────────────────────────────────────────────────────
    config["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
    )
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as f:
        yaml.safe_dump(config, f)

    # ── Loss functions ────────────────────────────────────────────────────────
    print_only(
        f"Instantiating Loss  train=<{config['loss']['train']['sdr_type']}>  "
        f"val=<{config['loss']['val']['sdr_type']}>"
    )
    loss_func = {
        "train": getattr(look2hear.losses, config["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    # ── Speaker contrastive params ────────────────────────────────────────────
    spk_cfg = config.get("spk_contrast", {})
    contrastive_weight = spk_cfg.get("loss_weight", 1.0)
    margin             = spk_cfg.get("margin", 0.0)
    alpha_pos          = spk_cfg.get("alpha_pos", 1.0)
    alpha_neg          = spk_cfg.get("alpha_neg", 1.0)
    beta_init          = spk_cfg.get("beta_init", 0.0)
    beta_final         = spk_cfg.get("beta_final", 1.5)
    beta_anneal_epochs = spk_cfg.get("beta_anneal_epochs", 100)

    print_only(
        f"[NASS v2] weight={contrastive_weight} margin={margin} "
        f"alpha_pos={alpha_pos} alpha_neg={alpha_neg} "
        f"beta={beta_init}→{beta_final} over {beta_anneal_epochs} epochs"
    )

    # ── Lightning system ──────────────────────────────────────────────────────
    system = AudioVisualLightningModuleNASSv2(
        audio_model=model,
        video_model=video_model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
        contrastive_weight=contrastive_weight,
        margin=margin,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        beta_init=beta_init,
        beta_final=beta_final,
        beta_anneal_epochs=beta_anneal_epochs,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint = ModelCheckpoint(
        exp_dir,
        filename="{epoch}",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks = [checkpoint]
    if config["training"].get("early_stop"):
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))

    # ── Logger ────────────────────────────────────────────────────────────────
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    os.makedirs(os.path.join(logger_dir, config["exp"]["exp_name"]), exist_ok=True)
    

    # ── Trainer ───────────────────────────────────────────────────────────────
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=distributed_backend,
        strategy=DDPStrategy(find_unused_parameters=True),
        limit_train_batches=1.0,
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=True,
        num_sanity_val_steps=0,
    )

    # ── Optional resume from checkpoint ──────────────────────────────────────
    if ckpt_path and os.path.exists(ckpt_path):
        print_only(f"Resuming from checkpoint: {ckpt_path}")
        trainer.fit(system, ckpt_path=ckpt_path)
    else:
        trainer.fit(system)

    print_only("Finished Training")

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()
    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

    args = parser.parse_args()
    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic, ckpt_path=args.ckpt_path)
