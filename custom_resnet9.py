# 定义ResNet-9模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from math import log


class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()
        # input conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # residual block 1
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.relu2_1 = nn.ReLU(inplace=True)
        
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(32)
        
        self.shortcut2 = nn.Sequential()
        self.relu2 = nn.ReLU(inplace=True)

        # residual block 2
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.relu3_1 = nn.ReLU(inplace=True)
        
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(32)

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32)
        )
        self.relu3 = nn.ReLU(inplace=True)

        # residual block 3
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.relu4_1 = nn.ReLU(inplace=True)
        
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(64)

        self.shortcut4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64)
        )
        self.relu4 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))

        residual2 = self.shortcut2(x)
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.bn2_2(self.conv2_2(x))
        x = self.relu2(x + residual2)
        
        residual3 = self.shortcut3(x)
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.bn3_2(self.conv3_2(x))
        x = self.relu3(x + residual3)
        
        residual4 = self.shortcut4(x)
        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = self.bn4_2(self.conv4_2(x))
        x = self.relu4(x + residual4)
        
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

if __name__ == "__main__":
    net = ResNet9(10)
    traced_model = torch.fx.symbolic_trace(net)
    print(traced_model.graph)
    
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(traced_model, dummy_input, "resnet9.onnx", opset_version=11)