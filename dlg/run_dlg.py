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


def get_optim(optim, dummy_data, dummy_label):
    lr = 1e-4
    params = [dummy_data, dummy_label]

    if optim == "LBFGS":
        optimizer = torch.optim.LBFGS(params)
        
    elif optim == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=lr)
        
    elif optim == "Adam":
        optimizer = torch.optim.Adam(params, lr=lr)
        
    elif optim == "SGD":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        
    elif optim == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=lr)
        
    elif optim == "Adadelta":
        optimizer = torch.optim.Adadelta(params, lr=lr)
        
    elif optim == "Adagrad":
        optimizer = torch.optim.Adagrad(params, lr=lr)
        
    elif optim == "Adamax":
        optimizer = torch.optim.Adamax(params, lr=lr)
        
    elif optim == "NAdam":
        optimizer = torch.optim.NAdam(params, lr=lr)
        
    elif optim == "RAdam":
        optimizer = torch.optim.RAdam(params, lr=lr)
        
    else:
        raise ValueError(f"Unknown optimizer: {optim}")
    
    return optimizer


def run_dlg(gt_data, gt_onehot_label, org_dx_dy, device, optim, net, criterion, epoch, global_precision):
    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device)

    if global_precision == 'float64':
        dummy_data = dummy_data.double()
        dummy_label = dummy_label.double()
    elif global_precision == 'float16':
        dummy_data = dummy_data.half()
        dummy_label = dummy_label.half()

    dummy_data.requires_grad_(True)
    dummy_label.requires_grad_(True)

    init_dummy_data = dummy_data.detach().clone()
    optimizer = get_optim(optim, dummy_data, dummy_label)

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
            for gx, gy in zip(dummy_dy_dx, org_dx_dy): 
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            
            return grad_diff
        
        #dummy_dy_dx가 origin_dy_dx와 비슷해지는 방향으로 step / dummy_data, dummy_label에 대해 step
        optimizer.step(closure)
        if iters % 10 == 0: 
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())

    return init_dummy_data, dummy_data
