import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict
import look2hear.losses
import look2hear.models
import look2hear.videomodels
import yaml
from rich import print
from ptflops import get_model_complexity_info

# # Set PyTorch thread count to 1
# torch.set_num_threads(1)

def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
)

with open("configs/Falcon.yml") as f:
    def_conf = yaml.safe_load(f)
parser = prepare_parser_from_dict(def_conf, parser=parser)
arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
# tmp = torch.cuda.memory_allocated(0)
audiomodel = getattr(look2hear.models, arg_dic["audionet"]["audionet_name"])(
   #sample_rate=arg_dic["datamodule"]["data_config"]["sample_rate"],
    **arg_dic["audionet"]["audionet_config"]
)
audiomodel.cuda()
# b = torch.cuda.memory_allocated(0)
# print((b - tmp) / 10 ** 6)
video_model = getattr(look2hear.videomodels, arg_dic["videonet"]["videonet_name"])(
        **arg_dic["videonet"]["videonet_config"],
)

# loss_func = getattr(look2hear.losses, arg_dic["loss"]["train"]["loss_func"])(
#             getattr(look2hear.losses, arg_dic["loss"]["train"]["sdr_type"]),
#             **arg_dic["loss"]["train"]["config"],
#         )
# optimizer = torch.optim.Adam(audiomodel.parameters(), lr=1e-4)
# with torch.cuda.device(3):
#     a = torch.randn(1, 16000).cuda()
#     v = torch.randn(1, 1, 25, 96, 96).cuda()
#     audiomodel = audiomodel.cuda()
#     video_model = video_model.cuda()
#     v = video_model(v)
#     print(audiomodel)
#     print(audiomodel(a, v).shape)
# print(v1 == v2)
# import pdb; pdb.set_trace()
# start = time.perf_counter()
# for i in range(100):
#     audiomodel(a)
# end = time.perf_counter()
# print((end - start)/100.)
def input_func(input):
    return {"audio_mixture":input[0],"mouth_embedding":input[1]}
with torch.cuda.device(0):
    video_model = video_model.cuda()
    video_model.eval()
    a = torch.randn(1, 32000).cuda()
    v = torch.randn(1, 1, 50, 64, 64).cuda()
    v = video_model(v)
    total_macs = 0
    total_params = 0
    # DPRNN
    model = audiomodel.cuda()
    model.eval()
    macs, params = get_model_complexity_info(
        model, 
        (a, v), 
        input_constructor=input_func, 
        #backend='aten',              # <--- add this line to use aten backend
        as_strings=False, 
        print_per_layer_stat=False, 
        verbose=False
    )
    # optimizer.zero_grad()
    # loss = loss_func(a.unsqueeze(0), model(a, v))
    # loss.backward()
    # optimizer.step()
    # for name, param in model.named_parameters():
    #     if param.requires_grad and param.grad is None:
    #         print(name)
    # model = audiomodel.cuda()
    # macs, params = get_model_complexity_info(
    #     model, (1,16000), as_strings=False, print_per_layer_stat=True, verbose=False
    # )
    total_macs += macs
    total_params += params
    # model = nn.Conv1d(1, 512, 64, 16).cuda()
    # macs, params = get_model_complexity_info(model, (1, 32000), as_strings=False,
    #                                                 print_per_layer_stat=True, verbose=False)
    # total_macs += macs
    # total_params += params

    # model = nn.ConvTranspose1d(512, 1, 64, 16).cuda()
    # macs, params = get_model_complexity_info(model, (512, 2000), as_strings=False,
    #                                                 print_per_layer_stat=True, verbose=False)
    # total_macs += macs*2
    # total_params += params*2
    # print(model(a, v).shape)
    print("MACs: ", total_macs / 10.0 ** 9)
    print("Params: ", total_params / 10.0 ** 6)
    
    # model.cpu()
    # a = a.cpu()
    # v = v.cpu()
    # # Set up cuda Events for timing measurement (PyTorch recommended interface)
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # # Initialize a timing container
    # timings = np.zeros((100, 1))

    # with torch.no_grad():
    #     for rep in tqdm(range(200)):
    #         if rep >= 100:
    #             starter.record()
    #         _ = model(a, v)
    #         if rep >= 100:
    #             ender.record()  # end timing
    #             torch.cuda.synchronize()  # wait for GPU tasks to complete
    #             timings[rep - 100] = starter.elapsed_time(ender)  # record time

    # avg = timings.sum()/100
    # print('\navg={}\n'.format(avg))

    # # Measure audio time
    # sample_rate = arg_dic["datamodule"]["data_config"]["sample_rate"]  # get sample rate from config
    # num_samples = a.shape[1]  # get number of audio samples

    # duration = num_samples / sample_rate  # compute duration (seconds)
    # print(f"Audio duration: {duration} seconds")
