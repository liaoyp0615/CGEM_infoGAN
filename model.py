# import ML package
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.tensor import Tensor
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Dict, Tuple, Any
from sklearn.utils import shuffle
import sys
import argparse
from torch.utils.data.dataset import Dataset
import uproot
import random

class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()

    def forward(self, x):
        x = 2 * F.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim

        self.init_size = opt.img_size // 4 # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128*self.init_size**3))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size, self.init_size)
        hist = self.conv_blocks(out)
        return hist


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [   nn.Conv3d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True)]
                        #nn.Dropout3d(0.25)]
            if bn:
                block.append(nn.BatchNorm3d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**3, 1))
        self.aux_layer = nn.Sequential(
            nn.Linear(128*ds_size**3, opt.n_classes),
            nn.Softmax()
        )
        self.latent_layer = nn.Sequential(nn.Linear(128*ds_size**3, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code
