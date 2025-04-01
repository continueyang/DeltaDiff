# Modified from https://github.com/JingyunLiang/SwinIR
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F

from basicsr.archs.swinir_arch import SwinIR


def maininf(img):

    scale=4
    model_path='weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth'



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(model_path)
    model.eval()
    model = model.to(device)


    window_size = 8

    img = img.to(device)
        # read image

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = img.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        output = model(img)
        _, _, h, w = output.size()
        output = output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

        return output

def define_model(path):
    # 001 classical image sr

    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=48,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv')



    loadnet = torch.load(path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


