# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
import yaml

print(torch.__version__, torchvision.__version__)

from utils import *

def main(config, args):
    # init
    torch.manual_seed(1234)
    tt = transforms.ToPILImage()
    dst = datasets.CIFAR100("./data/torch", download=True)
    device = set_device()
    criterion = cross_entropy_for_onehot

    # load config
    img_index = args.index
    batch_size = config['batch_size']
    epoch = config['epoch']

    # load data    
    gt_data, gt_label, gt_onehot_label = make_batched_input(dst, batch_size, img_index)
    gt_data = gt_data.to(device)
    gt_label = gt_label.to(device)
    gt_onehot_label = gt_onehot_label.to(device)

    # create model
    net = get_model(config['model'], device)

    # compute original gradient 
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    
    init_dummy_data = dummy_data.detach().clone()

    
    # create optim
    optimizer = get_optim(config['optim'], dummy_data, dummy_label)

    for iters in range(epoch):
        def closure():
            optimizer.zero_grad()

            #dummy forward
            dummy_pred = net(dummy_data) 
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            
            # dummy backward, model backward
            # dummy_dy_dx가 origin_dy_dx와 비슷해지는 방향으로 step하도록함
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            
            return grad_diff
        
        #dummy_dy_dx가 origin_dy_dx와 비슷해지는 방향으로 step / dummy_data, dummy_label에 대해 step
        optimizer.step(closure)
        if iters % 10 == 0: 
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())

    #show
    plt.figure(figsize=(2 * batch_size, 6))
    
    for i in range(batch_size):
        plt.subplot(3, batch_size, i + 1)
        plt.imshow(tt(gt_data[i].cpu()))
        plt.axis('off')
        if i == 0: plt.title("Original")

        plt.subplot(3, batch_size, batch_size + i + 1)
        plt.imshow(tt(init_dummy_data[i].cpu()))
        plt.axis('off')
        if i == 0: plt.title("Initial")

        plt.subplot(3, batch_size, 2 * batch_size + i + 1)
        plt.imshow(tt(dummy_data[i].detach().cpu()))
        plt.axis('off')
        if i == 0: plt.title("Final")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
    parser.add_argument('--index', type=int, default=400,
                        help='the index for leaking images on CIFAR.')
    parser.add_argument('--config', type=str, default="./config.yaml",
                        help='the path to yaml config file.')
    args = parser.parse_args()


    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    main(config, args)