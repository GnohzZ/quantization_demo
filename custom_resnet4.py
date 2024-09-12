# 定义ResNet-9模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from math import log


class ResNet4(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet4, self).__init__()
        # input conv
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        # residual block 1
        self.conv2_1 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.relu2_1 = nn.ReLU(inplace=True)
        
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(16)
        
        self.shortcut2 = nn.Conv2d(16, 16, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.relu2 = nn.ReLU(inplace=True)
        
        # 8x8x16
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))

        residual2 = self.bn2(self.shortcut2(x))
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.bn2_2(self.conv2_2(x))
        x = self.relu2(x + residual2)
        
        x = self.bn3(self.conv3(x))
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out
    

if __name__ == "__main__":
    net = ResNet4()
    rand_data = torch.randn(1, 3, 32, 32)
    out = net(rand_data)
    print(out.size())