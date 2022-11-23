#!/usr/bin/python3
#coding=utf-8

import os
import sys
#sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import Summar
# yWriter
from data import dataset
from net import MINet_ResNet50
import time
from PIL import Image
from tqdm import tqdm
import logging as logger
import torchvision

TAG = "WSLNet"
SAVE_PATH = TAG
GPU_ID=0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="test_%s.log"%(TAG), filemode="w")


DATASETS = ['./data_test/THUR', './data_test/PASCAL-S', './data_test/ECSSD', './data_test/HKU-IS', './data_test/DUTS-TE']


class Test(object):
    def __init__(self, Dataset, datapath, Network):
        ## dataset
        self.datapath = datapath.split("/")[-1]
        print("Testing on %s"%self.datapath)
        self.cfg = Dataset.Config(datapath = datapath, snapshot=sys.argv[1], mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=True, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.net.eval()
        self.to_pil = torchvision.transforms.ToPILImage()

    def _test_process(self, save_pre):
        loader = self.loader

        tqdm_iter = tqdm(enumerate(loader), total=len(loader), leave =False)
        for test_batch_id, test_data in tqdm_iter:
            tqdm_iter.set_description(f"{self.cfg.datapath.split('/')[-1]}: te=>{test_batch_id + 1}")
            with torch.no_grad():
                image, mask, _, (H, W), name = test_data
                image, mask            = image.cuda().float(), mask.cuda().float()
                outputs = self.net(image)

            outputs_np = outputs.sigmoid().cpu().detach()

            for out_item in outputs_np:
                gimg_path = os.path.join(self.cfg.datapath, 'image', name[0].split('.')[0] + '.jpg')
                gt_img = Image.open(gimg_path).convert("L")
                out_item[out_item<70/255] = 0
                out_item[out_item>200/255] = 1
                out_img = self.to_pil(out_item).resize(gt_img.size, resample=Image.NEAREST)

                if save_pre:
                    oimg_path = os.path.join('./pred_maps/{}/'.format(TAG) , self.cfg.datapath.split('/')[-1])
                    if not os.path.exists(oimg_path):
                        os.makedirs(oimg_path)
                    out_img.save(oimg_path + "/" + name[0])

if __name__=='__main__':
    for e in DATASETS:
        t =Test(dataset, e, MINet_ResNet50)
        t._test_process(True)

