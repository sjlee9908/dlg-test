import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        # CIFAR-100: 3 channels * 32 width * 32 height = 3072 inputs
        # 3 Layers: (Input->Hidden1) -> (Hidden1->Hidden2) -> (Hidden2->Output)
        
        self.fc1 = nn.Linear(3072, 1024) 
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 100) # CIFAR-100 classes

        # DLG 복원 시에는 ReLU보다 Sigmoid가 Gradient가 0이 되는 구간이 없어 유리함
        self.act = nn.Sigmoid() 

    def forward(self, x):
        # [Batch, 3, 32, 32] -> [Batch, 3072] 로 펼치기
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x) # 마지막은 Logits 출력 (Softmax는 Loss 계산 시 적용)
        return x