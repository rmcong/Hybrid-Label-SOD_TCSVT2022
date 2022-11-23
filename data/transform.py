#!/usr/bin/python3
#coding=utf-8

import cv2
import torch
import numpy as np
import random

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, mask, coarse):
        for op in self.ops:
            image, mask, coarse = op(image, mask, coarse)
        return image, mask, coarse

class RGBDCompose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, depth, mask):
        for op in self.ops:
            image, depth, mask = op(image, depth, mask)
        return image, depth, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask, coarse):
        image = (image - self.mean)/self.std
        mask /= 255
        coarse /= 255
        return image, mask, coarse

class RGBDNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, depth, mask):
        image = (image - self.mean)/self.std
        depth = (depth - self.mean) / self.std
        mask /= 255
        return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, coarse):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        coarse = cv2.resize(coarse, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, coarse

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, coarse):
        H,W,_ = image.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        image = image[ymin:ymin+self.H, xmin:xmin+self.W, :]
        mask  = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        coarse = coarse[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return image, mask, coarse

class RandomHorizontalFlip(object):
    def __call__(self, image, mask, coarse):
        if np.random.randint(2)==1:
            image = image[:,::-1,:].copy()
            mask  =  mask[:,::-1,:].copy()
            coarse = coarse[:,::-1,:].copy()
        return image, mask, coarse

class ToTensor(object):
    def __call__(self, image, mask, coarse):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        mask  = mask.permute(2, 0, 1)
        coarse = torch.from_numpy(coarse)
        coarse = coarse.permute(2, 0, 1)
        return image, mask.mean(dim=0, keepdim=True), coarse.mean(dim=0, keepdim=True)

class RandomRotateOrthogonal(object):
    def __call__(self, image, mask, coarse):
        if random.random() < 0.31:
            image  = cv2.flip(image, 0)
            mask   = cv2.flip(mask, 0)
            coarse = cv2.flip(coarse, 0)

        return image, mask, coarse

