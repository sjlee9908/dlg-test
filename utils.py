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


def make_batched_input(dataset, config, device):
    batch_size = config.batch_size
    start_idx = config.idx
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

    return gt_data.to(device), gt_label.to(device), gt_onehot_label.to(device)


def get_model(model, device):
    # create, init global model
    if model == "lenet":
        from models.lenet import LeNet, weights_init
        net = LeNet().to(device)
        net.apply(weights_init)
    
    elif model == "resnet":
        from models.resnet import resnet18, weights_init
        net = resnet18().to(device)
        net.apply(weights_init)
    
    elif model == "vggnet":
        from models.vgg import vgg11_bn, weights_init
        net = vgg11_bn().to(device)
        net.apply(weights_init)
    
    elif model == "ffn":
        from models.ffn import FFN, weights_init
        net = FFN().to(device)
        net.apply(weights_init)

    return net


import os
import matplotlib.pyplot as plt
from torchvision import transforms

def save_plot(save_path, batch_size, gt_data, init_dummy_data, dummy_data):
    """
    입력받은 save_path(경로+파일명)에 이미지를 저장합니다.
    구조 변경: (Original, Initial, Final)을 가로로 나란히 배치
    """
    
    # 1. 경로(Folder) 확인 및 생성
    dir_name = os.path.dirname(save_path) 
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        print(f"Directory created: {dir_name}")

    # 2. Plotting 로직 (가로 배치로 수정됨)
    tt = transforms.ToPILImage()
    
    # figsize 수정: 가로 폭은 고정(3개 이미지), 세로 길이는 batch_size에 비례
    # 예: 이미지 하나당 3x3인치라고 가정하면 가로 9, 세로 3*batch_size
    plt.figure(figsize=(9, 3 * batch_size))
    
    for i in range(batch_size):
        # --- Original (좌측) ---
        # 전체 행: batch_size, 전체 열: 3
        # 위치: 현재 행(i)의 첫 번째 칸 -> 3*i + 1
        plt.subplot(batch_size, 3, 3 * i + 1)
        plt.imshow(tt(gt_data[i].cpu()))
        plt.axis('off')
        if i == 0: plt.title("Original") # 맨 윗줄에만 타이틀 표시

        # --- Initial (중앙) ---
        # 위치: 현재 행(i)의 두 번째 칸 -> 3*i + 2
        plt.subplot(batch_size, 3, 3 * i + 2)
        plt.imshow(tt(init_dummy_data[i].cpu()))
        plt.axis('off')
        if i == 0: plt.title("Initial")

        # --- Final (우측) ---
        # 위치: 현재 행(i)의 세 번째 칸 -> 3*i + 3
        plt.subplot(batch_size, 3, 3 * i + 3)
        plt.imshow(tt(dummy_data[i].detach().cpu()))
        plt.axis('off')
        if i == 0: plt.title("Final")

    plt.tight_layout()
    
    # 3. 저장
    plt.savefig(save_path)
    plt.close()
    print(f"Saved image to: {save_path}")


def perceptual_similarity(gt_data, dlg_data, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models

    def infer(data):
        data = data.float()
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        upsampled = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
        normalized = (upsampled - mean) / std
        
        return vgg(normalized)

    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
    vgg.to(device)
    vgg.eval()

    gt_data = gt_data
    with torch.no_grad():
        gt_vector = infer(gt_data)
        dlg_vector = infer(dlg_data)
        per_sample_sim = F.cosine_similarity(gt_vector, dlg_vector, dim=1)
        final_score = per_sample_sim.mean().item()

    return final_score