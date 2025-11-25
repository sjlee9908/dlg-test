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


def make_batched_input(dataset, precision, batch_size, start_idx):
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

    if precision == 'float64':
        gt_data = gt_data.double()
        gt_onehot_label = gt_onehot_label.double()
    elif precision == 'float16':
        gt_data = gt_data.half()
        gt_onehot_label = gt_onehot_label.half()

    return gt_data, gt_label, gt_onehot_label


def get_model(model, precision, device):
    # create, init global model
    if model == "lenet":
        from models.lenet import LeNet, weights_init
        net = LeNet().to(device)
        net.apply(weights_init)
    
    if precision == "float16":
        net.half()  # 모델 전체를 float16으로 변환
    elif precision == "float64":
        net.double() 

    return net


import os
import matplotlib.pyplot as plt
from torchvision import transforms

def save_plot(save_path, batch_size, gt_data, init_dummy_data, dummy_data):
    """
    입력받은 save_path(경로+파일명)에 이미지를 저장합니다.
    경로상의 폴더가 없다면 자동으로 생성합니다.
    """
    
    # 1. 경로(Folder) 확인 및 생성 로직
    # 파일 경로에서 디렉토리 부분만 추출 (예: 'result/data/img.png' -> 'result/data')
    dir_name = os.path.dirname(save_path) 
    
    # 디렉토리가 명시되어 있고, 실제로 존재하지 않는다면 생성
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True) # exist_ok=True: 이미 있어도 에러 안 남
        print(f"Directory created: {dir_name}")

    # 2. 기존 Plotting 로직
    tt = transforms.ToPILImage()
    plt.figure(figsize=(2 * batch_size, 6))
    
    for i in range(batch_size):
        # Original
        plt.subplot(3, batch_size, i + 1)
        plt.imshow(tt(gt_data[i].cpu()))
        plt.axis('off')
        if i == 0: plt.title("Original")

        # Initial
        plt.subplot(3, batch_size, batch_size + i + 1)
        plt.imshow(tt(init_dummy_data[i].cpu()))
        plt.axis('off')
        if i == 0: plt.title("Initial")

        # Final
        plt.subplot(3, batch_size, 2 * batch_size + i + 1)
        plt.imshow(tt(dummy_data[i].detach().cpu()))
        plt.axis('off')
        if i == 0: plt.title("Final")

    plt.tight_layout()
    
    # 3. 저장 (show 대신 savefig 사용)
    plt.savefig(save_path)
    plt.close() # 메모리 해제
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