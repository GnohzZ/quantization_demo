import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from math import log
from quantization_utils import *
from ppq_onnx2torch import extractInt8QuantizedOnnx


class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()
        # input conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # residual block 1
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.relu2_1 = nn.ReLU(inplace=True)
        
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(32)
        
        self.shortcut2 = nn.Sequential()
        self.relu2 = nn.ReLU(inplace=True)

        # residual block 2
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.relu3_1 = nn.ReLU(inplace=True)
        
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_2 = nn.BatchNorm2d(32)

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32)
        )
        self.relu3 = nn.ReLU(inplace=True)

        # residual block 3
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.relu4_1 = nn.ReLU(inplace=True)
        
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4_2 = nn.BatchNorm2d(64)

        self.shortcut4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(64)
        )
        self.relu4 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # no bn
        # input conv
        x = FakeQuantize.apply(x, 32)
        
        x = self.conv1(x)
        x = self.relu1(x)
        
        # residual block 1
        residual2 = FakeQuantize.apply(self.shortcut2(x), 8)
        
        x = FakeQuantize.apply(x, 32)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = FakeQuantize.apply(x, 32)
        x = self.conv2_2(x)
        x = FakeQuantize.apply(x, 8)
        x = x + residual2

        x = FakeQuantize.apply(x, 16)
        x = self.relu2(x)
        
        # residual block 2
        residual3 = self.shortcut3(x)
        residual3 = FakeQuantize.apply(residual3, 8)
        
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = FakeQuantize.apply(x, 32)
        x = self.conv3_2(x)
        x = FakeQuantize.apply(x, 8)
        x = x + residual3
        x = FakeQuantize.apply(x, 16)
        x = self.relu3(x)
        
        # residual block 3
        residual4 = self.shortcut4(x)
        residual4 = FakeQuantize.apply(residual4, 4)
        
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = FakeQuantize.apply(x, 32)
        x = self.conv4_2(x)
        x = FakeQuantize.apply(x, 4)
        x = x + residual4
        x = FakeQuantize.apply(x, 8)
        x = self.relu4(x)
        
        x = self.avg_pool(x)
        x = FakeQuantize.apply(x, 32)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        x = FakeQuantize.apply(x, 8)
        
        return x
    
    def add_quantizer(self):
        parametrize.register_parametrization(self.conv1, "weight", WeightQuantizer(64))
        parametrize.register_parametrization(self.conv1, "bias", BiasQuantizer(64 * 32))
        
        parametrize.register_parametrization(self.conv2_1, "weight", WeightQuantizer(128))
        parametrize.register_parametrization(self.conv2_1, "bias", BiasQuantizer(128 * 32))
        
        parametrize.register_parametrization(self.conv2_2, "weight", WeightQuantizer(64))
        parametrize.register_parametrization(self.conv2_2, "bias", BiasQuantizer(64 * 32))
        
        parametrize.register_parametrization(self.conv3_1, "weight", WeightQuantizer(256))
        parametrize.register_parametrization(self.conv3_1, "bias", BiasQuantizer(256 * 16))
        
        parametrize.register_parametrization(self.conv3_2, "weight", WeightQuantizer(128))
        parametrize.register_parametrization(self.conv3_2, "bias", BiasQuantizer(128 * 32))
        
        parametrize.register_parametrization(self.shortcut3[0], "weight", WeightQuantizer(128))
        parametrize.register_parametrization(self.shortcut3[0], "bias", BiasQuantizer(128 * 16))
        
        parametrize.register_parametrization(self.conv4_1, "weight", WeightQuantizer(512))
        parametrize.register_parametrization(self.conv4_1, "bias", BiasQuantizer(512 * 16))
        
        parametrize.register_parametrization(self.conv4_2, "weight", WeightQuantizer(64))
        parametrize.register_parametrization(self.conv4_2, "bias", BiasQuantizer(64 * 32))
        
        parametrize.register_parametrization(self.shortcut4[0], "weight", WeightQuantizer(128))
        parametrize.register_parametrization(self.shortcut4[0], "bias", BiasQuantizer(128 * 16))
        
        parametrize.register_parametrization(self.fc, "weight", WeightQuantizer(64))
        parametrize.register_parametrization(self.fc, "bias", BiasQuantizer(64 * 32))

    def getParams(self, layer_info):
        self.conv1.weight.data = torch.Tensor(layer_info["/conv1/Conv"]["weight"])
        self.conv1.bias.data = torch.Tensor(layer_info["/conv1/Conv"]["bias"])
        
        self.conv2_1.weight.data = torch.Tensor(layer_info["/conv2_1/Conv"]["weight"])
        self.conv2_1.bias.data = torch.Tensor(layer_info["/conv2_1/Conv"]["bias"])
        
        self.conv2_2.weight.data = torch.Tensor(layer_info["/conv2_2/Conv"]["weight"])
        self.conv2_2.bias.data = torch.Tensor(layer_info["/conv2_2/Conv"]["bias"])
        
        self.conv3_1.weight.data = torch.Tensor(layer_info["/conv3_1/Conv"]["weight"])
        self.conv3_1.bias.data = torch.Tensor(layer_info["/conv3_1/Conv"]["bias"])
        
        self.conv3_2.weight.data = torch.Tensor(layer_info["/conv3_2/Conv"]["weight"])
        self.conv3_2.bias.data = torch.Tensor(layer_info["/conv3_2/Conv"]["bias"])
        
        self.shortcut3[0].weight.data = torch.Tensor(layer_info["/shortcut3/shortcut3.0/Conv"]["weight"])
        self.shortcut3[0].bias.data = torch.Tensor(layer_info["/shortcut3/shortcut3.0/Conv"]["bias"])
        
        self.conv4_1.weight.data = torch.Tensor(layer_info["/conv4_1/Conv"]["weight"])
        self.conv4_1.bias.data = torch.Tensor(layer_info["/conv4_1/Conv"]["bias"])
        
        self.conv4_2.weight.data = torch.Tensor(layer_info["/conv4_2/Conv"]["weight"])
        self.conv4_2.bias.data = torch.Tensor(layer_info["/conv4_2/Conv"]["bias"])
        
        self.shortcut4[0].weight.data = torch.Tensor(layer_info["/shortcut4/shortcut4.0/Conv"]["weight"])
        self.shortcut4[0].bias.data = torch.Tensor(layer_info["/shortcut4/shortcut4.0/Conv"]["bias"])
        
        self.fc.weight.data = torch.Tensor(layer_info["/fc/Gemm"]["weight"])
        self.fc.bias.data = torch.Tensor(layer_info["/fc/Gemm"]["bias"])        
    

class QuantizedResNet9(nn.Module):
    def __init__(self, layer_info, num_classes=10):
        super(QuantizedResNet9, self).__init__()
        self.layer_info = layer_info
        
        # input conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        # residual block 1
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.shortcut2 = nn.Sequential()
        self.relu2 = nn.ReLU(inplace=True)

        # residual block 2
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0, bias=True),
        )
        self.relu3 = nn.ReLU(inplace=True)

        # residual block 3
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.shortcut4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=True),
        )
        self.relu4 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def getParams(self):
        layer_info = self.layer_info
        
        self.conv1.weight.data = torch.Tensor(layer_info["/conv1/Conv"]["weight"])
        self.conv1.bias.data = torch.Tensor(layer_info["/conv1/Conv"]["bias"])
        
        self.conv2_1.weight.data = torch.Tensor(layer_info["/conv2_1/Conv"]["weight"])
        self.conv2_1.bias.data = torch.Tensor(layer_info["/conv2_1/Conv"]["bias"])
        
        self.conv2_2.weight.data = torch.Tensor(layer_info["/conv2_2/Conv"]["weight"])
        self.conv2_2.bias.data = torch.Tensor(layer_info["/conv2_2/Conv"]["bias"])
        
        self.conv3_1.weight.data = torch.Tensor(layer_info["/conv3_1/Conv"]["weight"])
        self.conv3_1.bias.data = torch.Tensor(layer_info["/conv3_1/Conv"]["bias"])
        
        self.conv3_2.weight.data = torch.Tensor(layer_info["/conv3_2/Conv"]["weight"])
        self.conv3_2.bias.data = torch.Tensor(layer_info["/conv3_2/Conv"]["bias"])
        
        self.shortcut3[0].weight.data = torch.Tensor(layer_info["/shortcut3/shortcut3.0/Conv"]["weight"])
        self.shortcut3[0].bias.data = torch.Tensor(layer_info["/shortcut3/shortcut3.0/Conv"]["bias"])
        
        self.conv4_1.weight.data = torch.Tensor(layer_info["/conv4_1/Conv"]["weight"])
        self.conv4_1.bias.data = torch.Tensor(layer_info["/conv4_1/Conv"]["bias"])
        
        self.conv4_2.weight.data = torch.Tensor(layer_info["/conv4_2/Conv"]["weight"])
        self.conv4_2.bias.data = torch.Tensor(layer_info["/conv4_2/Conv"]["bias"])
        
        self.shortcut4[0].weight.data = torch.Tensor(layer_info["/shortcut4/shortcut4.0/Conv"]["weight"])
        self.shortcut4[0].bias.data = torch.Tensor(layer_info["/shortcut4/shortcut4.0/Conv"]["bias"])
        
        self.fc.weight.data = torch.Tensor(layer_info["/fc/Gemm"]["weight"])
        self.fc.bias.data = torch.Tensor(layer_info["/fc/Gemm"]["bias"])
    
    def quantize(self):
        self.conv1.weight.data = (self.conv1.weight.data / self.layer_info["/conv1/Conv"]["weight_scale"] + self.layer_info["/conv1/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv1.bias.data = (self.conv1.bias.data / self.layer_info["/conv1/Conv"]["bias_scale"] + self.layer_info["/conv1/Conv"]["bias_zero_point"]).round()
        
        self.conv2_1.weight.data = (self.conv2_1.weight.data / self.layer_info["/conv2_1/Conv"]["weight_scale"] + self.layer_info["/conv2_1/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv2_1.bias.data = (self.conv2_1.bias.data / self.layer_info["/conv2_1/Conv"]["bias_scale"] + self.layer_info["/conv2_1/Conv"]["bias_zero_point"]).round()
        
        self.conv2_2.weight.data = (self.conv2_2.weight.data / self.layer_info["/conv2_2/Conv"]["weight_scale"] + self.layer_info["/conv2_2/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv2_2.bias.data = (self.conv2_2.bias.data / self.layer_info["/conv2_2/Conv"]["bias_scale"] + self.layer_info["/conv2_2/Conv"]["bias_zero_point"]).round()
        
        self.conv3_1.weight.data = (self.conv3_1.weight.data / self.layer_info["/conv3_1/Conv"]["weight_scale"] + self.layer_info["/conv3_1/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv3_1.bias.data = (self.conv3_1.bias.data / self.layer_info["/conv3_1/Conv"]["bias_scale"] + self.layer_info["/conv3_1/Conv"]["bias_zero_point"]).round()
        
        self.conv3_2.weight.data = (self.conv3_2.weight.data / self.layer_info["/conv3_2/Conv"]["weight_scale"] + self.layer_info["/conv3_2/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv3_2.bias.data = (self.conv3_2.bias.data / self.layer_info["/conv3_2/Conv"]["bias_scale"] + self.layer_info["/conv3_2/Conv"]["bias_zero_point"]).round()
        
        self.shortcut3[0].weight.data = (self.shortcut3[0].weight.data / self.layer_info["/shortcut3/shortcut3.0/Conv"]["weight_scale"] + self.layer_info["/shortcut3/shortcut3.0/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.shortcut3[0].bias.data = (self.shortcut3[0].bias.data / self.layer_info["/shortcut3/shortcut3.0/Conv"]["bias_scale"] + self.layer_info["/shortcut3/shortcut3.0/Conv"]["bias_zero_point"]).round()
        
        self.conv4_1.weight.data = (self.conv4_1.weight.data / self.layer_info["/conv4_1/Conv"]["weight_scale"] + self.layer_info["/conv4_1/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv4_1.bias.data = (self.conv4_1.bias.data / self.layer_info["/conv4_1/Conv"]["bias_scale"] + self.layer_info["/conv4_1/Conv"]["bias_zero_point"]).round()
        
        self.conv4_2.weight.data = (self.conv4_2.weight.data / self.layer_info["/conv4_2/Conv"]["weight_scale"] + self.layer_info["/conv4_2/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv4_2.bias.data = (self.conv4_2.bias.data / self.layer_info["/conv4_2/Conv"]["bias_scale"] + self.layer_info["/conv4_2/Conv"]["bias_zero_point"]).round()
        
        self.shortcut4[0].weight.data = (self.shortcut4[0].weight.data / self.layer_info["/shortcut4/shortcut4.0/Conv"]["weight_scale"] + self.layer_info["/shortcut4/shortcut4.0/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.shortcut4[0].bias.data = (self.shortcut4[0].bias.data / self.layer_info["/shortcut4/shortcut4.0/Conv"]["bias_scale"] + self.layer_info["/shortcut4/shortcut4.0/Conv"]["bias_zero_point"]).round()
        
        self.fc.weight.data = (self.fc.weight.data / self.layer_info["/fc/Gemm"]["weight_scale"] + self.layer_info["/fc/Gemm"]["weight_zero_point"]).round().clamp(-128, 127)        
        self.fc.bias.data = (self.fc.bias.data / self.layer_info["/fc/Gemm"]["bias_scale"] + self.layer_info["/fc/Gemm"]["bias_zero_point"]).round()
        
    
    def forward(self, x):
        # no bn
        # input conv
        x = (x / self.layer_info["/conv1/Conv"]["input_scale"]).round().clamp(-128, 127)
        
        x = self.conv1(x)
        x = self.relu1(x)
        
        # residual block 1
        residual2 = cut_scale(x, self.layer_info["/conv1/Conv"]["bias_scale"], self.layer_info["/Add"]["input_1_scale"])
        residual2 = self.shortcut2(residual2)
        
        x = cut_scale(x, self.layer_info["/conv1/Conv"]["bias_scale"], self.layer_info["/conv2_1/Conv"]["input_scale"])
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        
        x = cut_scale(x, self.layer_info["/conv2_1/Conv"]["bias_scale"], self.layer_info["/conv2_2/Conv"]["input_scale"])
        x = self.conv2_2(x)
        x = cut_scale(x, self.layer_info["/conv2_2/Conv"]["bias_scale"], self.layer_info["/Add"]["input_2_scale"])
        
        x = x + residual2
        x = cut_scale(x, self.layer_info["/Add"]["input_1_scale"], self.layer_info["/Add"]["output_scale"])
        x = self.relu2(x)
        
        # residual block 2
        residual3 = self.shortcut3(x)
        residual3 = cut_scale(residual3, self.layer_info["/shortcut3/shortcut3.0/Conv"]["bias_scale"], self.layer_info["/Add_1"]["input_1_scale"])
        
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = cut_scale(x, self.layer_info["/conv3_1/Conv"]["bias_scale"], self.layer_info["/conv3_2/Conv"]["input_scale"])
        
        x = self.conv3_2(x)
        x = cut_scale(x, self.layer_info["/conv3_2/Conv"]["bias_scale"], self.layer_info["/Add_1"]["input_2_scale"])
        
        x = x + residual3
        x = cut_scale(x, self.layer_info["/Add_1"]["input_1_scale"], self.layer_info["/Add_1"]["output_scale"])
        x = self.relu3(x)
        
        # residual block 3
        residual4 = self.shortcut4(x)
        residual4 = cut_scale(residual4, self.layer_info["/shortcut4/shortcut4.0/Conv"]["bias_scale"], self.layer_info["/Add_2"]["input_1_scale"])
        
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = cut_scale(x, self.layer_info["/conv4_1/Conv"]["bias_scale"], self.layer_info["/conv4_2/Conv"]["input_scale"])
        
        x = self.conv4_2(x)
        x = cut_scale(x, self.layer_info["/conv4_2/Conv"]["bias_scale"], self.layer_info["/Add_2"]["input_2_scale"])
        
        x = x + residual4
        x = cut_scale(x, self.layer_info["/Add_2"]["input_1_scale"], self.layer_info["/Add_2"]["output_scale"])
        x = self.relu4(x)
        
        x = self.avg_pool(x)
        
        x = cut_scale(x, self.layer_info["/Add_2"]["output_scale"], self.layer_info["/fc/Gemm"]["input_scale"])        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        x = cut_scale(x, self.layer_info["/fc/Gemm"]["bias_scale"], self.layer_info["/fc/Gemm"]["output_scale"])
        
        x = x * self.layer_info["/fc/Gemm"]["output_scale"]
        
        return x
        
    

if __name__ == "__main__":
    # net = ResNet9(10)
    # net.add_quantizer()
    
    net = QuantizedResNet9(extractInt8QuantizedOnnx("./ckpt/MQuantized.onnx"))
    net.quantize()
    print(net)