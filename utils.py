import torch
import torch.nn.functional as F

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



def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def set_device():
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    print("Running on %s" % device)
    return device


def make_batched_input(dataset, batch_size, start_idx):
    tp = transforms.ToTensor()

    gt_data_list = []
    gt_label_list = []
    for i in range(batch_size):
        current_idx = (start_idx + i) % len(dataset)
        data, label = dataset[current_idx]
        data = tp(data)
        gt_data_list.append(data)
        gt_label_list.append(torch.tensor(label))

    gt_data = torch.stack(gt_data_list)
    gt_label = torch.stack(gt_label_list).long()
    gt_onehot_label = label_to_onehot(gt_label, num_classes=100)

    return gt_data, gt_label, gt_onehot_label


def get_model(model, device):
        
    # create, init global model
    if model == "lenet":
        from models.lenet import LeNet, weights_init
        net = LeNet().to(device)
        net.apply(weights_init)
    elif model == 'resnet18':
        from models.resnet import resnet18, weights_init
        net = resnet18().to(device)
        net.apply(weights_init)
    elif model == 'resnet34':
        from models.resnet import resnet34, weights_init
        net = resnet34().to(device)
        net.apply(weights_init)
    elif model == 'resnet50':
        from models.resnet import resnet101, weights_init
        net = resnet101().to(device)
        net.apply(weights_init)
    elif model == 'resnet101':
        from models.resnet import resnet152, weights_init
        net = resnet152().to(device)
        net.apply(weights_init)
    elif model == 'resnet152':
        from models.resnet import resnet152, weights_init
        net = resnet152().to(device)
        net.apply(weights_init)
    
    return net

def get_optim(optim, dummy_data, dummy_label):
    lr = 1
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


# def ceptual_cos_sim():
