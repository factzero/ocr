# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from data.dataloader import ErasingData
from loss.Loss import LossWithGAN_STE
from models.sa_gan import STRnet2


parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0,
                    help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='',
                    help='path for saving models')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=512,
                    help='image loading size')
parser.add_argument('--dataRoot', type=str,
                    default='')
parser.add_argument('--pretrained',type=str, default='', help='pretrained models for finetuning')
parser.add_argument('--num_epochs', type=int, default=500, help='epochs')
args = parser.parse_args()


def visual(image):
    im = image.transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()


batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)

if not os.path.exists(args.modelsSavePath):
    os.makedirs(args.modelsSavePath)

dataRoot = args.dataRoot
Erase_data = ErasingData(dataRoot, loadSize, training=True)
Erase_data = DataLoader(Erase_data, batch_size=batchSize, 
                         shuffle=True, num_workers=args.numOfWorkers, drop_last=False, pin_memory=True)

netG = STRnet2(3)
if args.pretrained != '':
    print('loaded ')
    netG.load_state_dict(torch.load(args.pretrained))

G_optimizer = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.9))
criterion = LossWithGAN_STE()

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda is available!')
    cudnn.enable = True
    cudnn.benchmark = True
    netG = netG.cuda()
    criterion = criterion.cuda()

for i in range(1, args.num_epochs + 1):
    netG.train()
    loss = 0

    for k,(imgs, gt, masks, path) in enumerate(Erase_data):
        if cuda:
            imgs = imgs.cuda()
            gt = gt.cuda()
            masks = masks.cuda()
        netG.zero_grad()

        x_o1, x_o2, x_o3, fake_images, mm = netG(imgs)
        G_loss = criterion(masks, fake_images, mm, gt)
        G_loss = G_loss.sum()
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        loss = G_loss.item()      

        print('[{}/{}] Generator Loss of epoch{} is {}'.format(k, len(Erase_data), i, G_loss.item()))
    
    torch.save(netG.state_dict(), args.modelsSavePath + '/STE_{}_{}.pth'.format(i, loss))