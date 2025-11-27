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

from dlg.client_runner import ClientRunner 
from dlg.dlg_runner import DLGRunner

print(torch.__version__, torchvision.__version__)

from utils import *

def main(config, scen):
    dst = datasets.CIFAR100("./data/torch", download=True)
    device = set_device()

    gt_data, gt_label, gt_onehot_label = make_batched_input(
                                            dst,
                                            config = config.data,
                                            device = device)
    
    net = get_model(config['model'], device)

    client = ClientRunner(net, config.client)
    attacker = DLGRunner(
                    gt_data_size=gt_data.size(),
                    gt_onehot_label_size=gt_onehot_label.size(),
                    device=device,
                    config = config.dlg)
    
    dy_dx = client.run_client(gt_data, gt_onehot_label, cross_entropy_for_onehot)
    init_dd, dd = attacker.run_dlg(dy_dx, net, cross_entropy_for_onehot)

    sim = perceptual_similarity(gt_data, dd, device)

    with open(f"./result/{scen}_{config.data.idx}.txt", "a") as f:
        f.write(f"cosine sim : {sim}\n")
    save_plot(f"./result/{scen}_{config.data.idx}.png", config.data.batch_size, gt_data, init_dd, dd)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
    parser.add_argument('--config', type=str, default="./config.yaml",
                        help='the path to yaml config file.')
    parser.add_argument('--scenario', type=str, default="org",
                        help='scenario to run')
    args = parser.parse_args()

    torch.manual_seed(1234)

    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config)


    main(config.get(args.scenario), args.scenario)