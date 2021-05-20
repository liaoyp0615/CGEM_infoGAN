from __future__ import print_function
import argparse
from sklearn.utils import shuffle
import sys
import argparse
import logging
import os
import ast
import gc
import psutil
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import math
from torch.utils.data.dataset import Dataset
from dataset import *
#from model import *
import uproot
import random

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=62, help='dimensionality of the latent space')
parser.add_argument('--code_dim', type=int, default=2, help='latent code')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--datafile', type=str, help='ROOT file paths')
parser.add_argument('--num_hist', type=int, default=500, help='Number of histograms to train for.')
parser.add_argument('--D_restore_pkl_path', type=str, help='D restore_pkl_path pkl file paths')
parser.add_argument('--D_pkl_path', type=str, help='D pkl_path pkl file paths')
parser.add_argument('--G_restore_pkl_path', type=str, help='G restore_pkl_path pkl file paths')
parser.add_argument('--G_pkl_path', type=str, help='G pkl_path pkl file paths')
parser.add_argument('--restore', type=ast.literal_eval, default=False, help='ckpt file paths')
opt = parser.parse_args()
print(opt)

logger = logging.getLogger('%s.%s' % ( __package__, os.path.splitext(os.path.split(__file__)[-1])[0]))
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s' '[%(levelname)s]: %(message)s')
hander = logging.StreamHandler(sys.stdout)
hander.setFormatter(formatter)
logger.addHandler(hander)
logger.info('constructing graph')

cuda = True if torch.cuda.is_available() else False

#--------------
# Architecture
#--------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim

        self.l1 = nn.Sequential(
                    nn.Linear(input_dim, 2048),
                    nn.BatchNorm1d(2048),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(2048, 2*2*2*128),
                    nn.BatchNorm1d(2*2*2*128),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
        self.conv_blocks = nn.Sequential(
                    nn.ConvTranspose3d(128,64,3,1,0),
                    nn.BatchNorm3d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ConvTranspose3d(64,32,5,3,0),
                    nn.BatchNorm3d(32),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ConvTranspose3d(32,16,5,3,0),
                    nn.BatchNorm3d(16),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ConvTranspose3d(16,1,6,2,0),
                    nn.Tanh()
                    )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128,2,2,2)
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
            *discriminator_block(opt.channels, 16, bn=False), #92-46-23-11-6
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size / 2**4
        ds_size = math.ceil(ds_size)

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**3, 1))
        self.aux_layer = nn.Sequential(
            nn.Linear(128*ds_size**3, opt.n_classes),
            nn.Softmax()
        )
        self.latent_layer = nn.Sequential(nn.Linear(128*ds_size**3, opt.code_dim))

    def forward(self, img):
        print("1:",img.shape)
        out = self.conv_blocks(img)
        print("2:",out.shape)
        out = out.view(out.shape[0], -1)
        print("3:",out.shape)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.

    return Variable(FloatTensor(y_cat))

# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1

# Initialize generator and discriminator
G = Generator()
D = Discriminator()

if cuda:
    G.cuda()
    D.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()

# Initialize weights
if opt.restore:
    G.load_state_dict(torch.load(opt.G_restore_pkl_path))
    D.load_state_dict(torch.load(opt.D_restore_pkl_path))
    print('restored from ',opt.G_restore_pkl_path)
    print('restored from ',opt.D_restore_pkl_path)
else:
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(itertools.chain(G.parameters(), D.parameters()),
                                    lr=opt.lr, betas=(opt.b1, opt.b2))
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
num_load = 2

# ----------
#  Training
# ----------

for epoch in range( 1, opt.n_epochs+1 ):
    if epoch%(opt.n_epochs//num_load)==0 or epoch==1:
        #load the data.
        logger.info('Start loading data:')
        logger.info(opt.datafile)
        Dataset_Train, train_loader = None, None
        del Dataset_Train, train_loader
        gc.collect()
        Dataset_Train = HistoDataset(opt.datafile, opt.num_hist)
        train_loader = torch.utils.data.DataLoader(Dataset_Train, batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=0)
        logger.info('Load data successfully! Start training...')
    
    for batch_idx, data in enumerate(train_loader):
        batch_size = data.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake  = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_data = Variable(data.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes) # --------------
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        # Generate a batch of images
        fake_data = G(z, label_input, code_input)

        # Loss measures generator's ability to fool the discriminator
        validity, _, _ = D(fake_data)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, _, _ = D(real_data)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _, _ = D(fake_data.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()


        #------------------
        # Information Loss
        #------------------

        optimizer_info.zero_grad()

        # Sample labels
        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

        # Ground truth labels
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)


        # Sample noise, labels and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.normal(-1, 1, (batch_size, opt.code_dim))))

        gen_imgs = G(z, label_input, code_input)
        _, pred_label, pred_code = D(gen_imgs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + \
                    lambda_con * continuous_loss(pred_code, code_input)

        info_loss.backward()
        optimizer_info.step()

    if epoch == 1:
        logger.info("1 epoch completed! This code is running successfully!")
    if epoch%(opt.n_epochs//10)==0:
        logger.info( "Epoch %6d. D_Loss %5.3f. G_Loss %5.3f. info_Loss %5.3f." % ( epoch, d_loss, g_loss, info_loss ) )
    if epoch%200==0:
        torch.save(D.state_dict(), opt.D_pkl_path)
        torch.save(G.state_dict(), opt.G_pkl_path)
        logger.info('Model save into:')
        logger.info(opt.D_pkl_path)
        logger.info(opt.G_pkl_path)

logger.info("Train Done!")
