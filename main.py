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

from dlg.run_dlg import run_dlg

print(torch.__version__, torchvision.__version__)

from utils import *

def main(config, scen):
    pprint(config)
    # init
    dst = datasets.CIFAR100("./data/torch", download=True)
    device = set_device()
    criterion = cross_entropy_for_onehot

    # load config
    img_index = config['idx']

    # load data    
    gt_data, gt_label, gt_onehot_label = make_batched_input(dst, config["local_precision"], config['batch_size'], img_index)
    gt_data = gt_data.to(device)
    gt_label = gt_label.to(device)
    gt_onehot_label = gt_onehot_label.to(device)

    # create model
    net = get_model(config['model'], config["local_precision"], device)

    # compute original gradient 
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    
    if config['noise_level'] != 0:
        noise_ratio = config['noise_level'] * 0.1
        
        noisy_dy_dx = []
        for g in dy_dx:
            noise_scale = g.mean() * noise_ratio
            noise = torch.randn_like(g) * noise_scale
            noisy_dy_dx.append(g + noise)
            
        dy_dx = tuple(noisy_dy_dx)

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    if config['global_precision'] == "float64":
        original_dy_dx = [g.double() for g in original_dy_dx]
    elif config['global_precision'] == "float16":
        original_dy_dx = [g.half() for g in original_dy_dx]
    else:
        original_dy_dx = [g.float() for g in original_dy_dx]
    

    init_dummy_data, dummy_data = run_dlg(
        gt_data = gt_data, 
        gt_onehot_label = gt_onehot_label, 
        org_dx_dy = original_dy_dx, 
        device = device, 
        optim = config["optim"],
        net = net, 
        criterion = criterion, 
        epoch = config["epoch"],
        global_precision=config['global_precision']
        )
    
    sim = perceptual_similarity(gt_data, dummy_data, device)
    with open(f"./result/{scen}_{img_index}.txt", "a") as f:
        f.write(f"cosine sim : {sim}\n")  # 끝에 \n을 붙여야 줄바꿈이 됩니다.
    save_plot(f"./result/{scen}_{img_index}.png", config['batch_size'], gt_data, init_dummy_data, dummy_data)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
    parser.add_argument('--config', type=str, default="./config.yaml",
                        help='the path to yaml config file.')
    parser.add_argument('--scenario', type=str, default="org",
                        help='scenario to run')
    args = parser.parse_args()

    torch.manual_seed(1234)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    main(config[args.scenario], args.scenario)