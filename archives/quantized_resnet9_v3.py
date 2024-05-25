import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from math import log
from quantization_utils import FakeQuantize, WeightQuantizer, BiasQuantizer, cut
import logging

# 配置日志记录
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# 计算缩放因子和最接近的2的整次幂
def calculate_scale_factor(max_abs_value):
    scale_factor = 128 / max_abs_value
    return scale_factor

def nearest_power_of_2(value):
    log2 = log(value, 2)
    nearest_power = round(log2)
    return 2 ** nearest_power

# 定义ResNet-9模型
class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_shortcut1 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0, bias=True)

        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_shortcut2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=True)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.clamp(x, -1, 1)
        x = self.relu(x)

        residual1 = x
        out2_1 = self.conv2_1(x)
        out2_1 = torch.clamp(out2_1, -1, 1)
        out2_1 = self.relu2_1(out2_1)

        out2_2 = self.conv2_2(out2_1)
        out2_2 = torch.clamp(out2_2, -1, 1)
        
        out2_2 += residual1
        out2_2 = self.relu(out2_2)

        residual2 = self.conv_shortcut1(out2_2)
        residual2 = torch.clamp(residual2, -1, 1)
        out3_1 = self.conv3_1(out2_2)
        out3_1 = torch.clamp(out3_1, -1, 1)
        out3_1 = self.relu3_1(out3_1)
        
        out3_2 = self.conv3_2(out3_1)
        out3_2 = torch.clamp(out3_2, -1, 1)
        
        out3_2 += residual2
        out3_2 = self.relu(out3_2)

        residual3 = self.conv_shortcut2(out3_2)
        residual3 = torch.clamp(residual3, -1, 1)
        out4_1 = self.conv4_1(out3_2)
        out4_1 = torch.clamp(out4_1, -1, 1)
        out4_1 = self.relu4_1(out4_1)

        out4_2 = self.conv4_2(out4_1)
        out4_2 = torch.clamp(out4_2, -1, 1)
        out4_2 += residual3
        out4_2 = self.relu(out4_2)
        out4_2 = torch.clamp(out4_2, -1, 1)
        
        out = self.avg_pool(out4_2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = torch.clamp(out, -1, 1)
        return out

# 定义ResNet-9 QuantizAware模型
class ResNet9_QuantizAware(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet9_QuantizAware, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_shortcut1 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0, bias=True)

        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_shortcut2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=True)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def add_quantizer(self, scale_factors):
        parametrize.register_parametrization(self.conv1, "weight", WeightQuantizer(scale_factors['conv1.weight']))
        parametrize.register_parametrization(self.conv2_1, "weight", WeightQuantizer(scale_factors['conv2_1.weight']))
        parametrize.register_parametrization(self.conv2_2, "weight", WeightQuantizer(scale_factors['conv2_2.weight']))
        parametrize.register_parametrization(self.conv3_1, "weight", WeightQuantizer(scale_factors['conv3_1.weight']))
        parametrize.register_parametrization(self.conv3_2, "weight", WeightQuantizer(scale_factors['conv3_2.weight']))
        parametrize.register_parametrization(self.conv_shortcut1, "weight", WeightQuantizer(scale_factors['conv_shortcut1.weight']))
        parametrize.register_parametrization(self.conv4_1, "weight", WeightQuantizer(scale_factors['conv4_1.weight']))
        parametrize.register_parametrization(self.conv4_2, "weight", WeightQuantizer(scale_factors['conv4_2.weight']))
        parametrize.register_parametrization(self.conv_shortcut2, "weight", WeightQuantizer(scale_factors['conv_shortcut2.weight']))
        parametrize.register_parametrization(self.fc, "weight", WeightQuantizer(scale_factors['fc.weight']))
        
        parametrize.register_parametrization(self.conv1, "bias", BiasQuantizer(scale_factors['conv1.bias']))
        parametrize.register_parametrization(self.conv2_1, "bias", BiasQuantizer(scale_factors['conv2_1.bias']))
        parametrize.register_parametrization(self.conv2_2, "bias", BiasQuantizer(scale_factors['conv2_2.bias']))
        parametrize.register_parametrization(self.conv3_1, "bias", BiasQuantizer(scale_factors['conv3_1.bias']))
        parametrize.register_parametrization(self.conv3_2, "bias", BiasQuantizer(scale_factors['conv3_2.bias']))
        parametrize.register_parametrization(self.conv_shortcut1, "bias", BiasQuantizer(scale_factors['conv_shortcut1.bias']))
        parametrize.register_parametrization(self.conv4_1, "bias", BiasQuantizer(scale_factors['conv4_1.bias']))
        parametrize.register_parametrization(self.conv4_2, "bias", BiasQuantizer(scale_factors['conv4_2.bias']))
        parametrize.register_parametrization(self.conv_shortcut2, "bias", BiasQuantizer(scale_factors['conv_shortcut2.bias']))
        parametrize.register_parametrization(self.fc, "bias", BiasQuantizer(scale_factors['fc.bias']))

    def remove_quantizer(self):
        parametrize.remove_parametrizations(self.conv1, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv1, "bias", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv2_1, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv2_1, "bias", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv2_2, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv2_2, "bias", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv3_1, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv3_1, "bias", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv3_2, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv3_2, "bias", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv_shortcut1, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv_shortcut1, "bias", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv4_1, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv4_1, "bias", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv4_2, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv4_2, "bias", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv_shortcut2, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(self.conv_shortcut2, "bias", leave_parametrized=False)
        parametrize.remove_parametrizations(self.fc, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(self.fc, "bias", leave_parametrized=False)

    def forward(self, x):
        x = FakeQuantize.apply(x, 128)
        x = self.conv1(x)
        x = FakeQuantize.apply(x, 128)
        x = torch.clamp(x, -1, 1)
        x = self.relu(x)

        residual1 = x
        out2_1 = self.conv2_1(x)
        out2_1 = FakeQuantize.apply(out2_1, 128)
        out2_1 = torch.clamp(out2_1, -1, 1)
        out2_1 = self.relu2_1(out2_1)
        out2_2 = self.conv2_2(out2_1)
        out2_2 = FakeQuantize.apply(out2_2, 128)
        out2_2 = torch.clamp(out2_2, -1, 1)
        out2_2 += residual1
        out2_2 = FakeQuantize.apply(out2_2, 128)
        out2_2 = torch.clamp(out2_2, -1, 1)
        out2_2 = self.relu(out2_2)

        residual2 = self.conv_shortcut1(out2_2)
        residual2 = FakeQuantize.apply(residual2, 128)
        residual2 = torch.clamp(residual2, -1, 1)
        out3_1 = self.conv3_1(out2_2)
        out3_1 = FakeQuantize.apply(out3_1, 128)
        out3_1 = torch.clamp(out3_1, -1, 1)
        out3_1 = self.relu3_1(out3_1)
        out3_2 = self.conv3_2(out3_1)
        out3_2 = FakeQuantize.apply(out3_2, 128)
        out3_2 = torch.clamp(out3_2, -1, 1)
        out3_2 += residual2
        out3_2 = FakeQuantize.apply(out3_2, 128)
        out3_2 = torch.clamp(out3_2, -1, 1)
        out3_2 = self.relu(out3_2)

        residual3 = self.conv_shortcut2(out3_2)
        residual3 = FakeQuantize.apply(residual3, 128)
        residual3 = torch.clamp(residual3, -1, 1)
        out4_1 = self.conv4_1(out3_2)
        out4_1 = FakeQuantize.apply(out4_1, 128)
        out4_1 = torch.clamp(out4_1, -1, 1)
        out4_1 = self.relu4_1(out4_1)
        out4_2 = self.conv4_2(out4_1)
        out4_2 = FakeQuantize.apply(out4_2, 128)
        out4_2 = torch.clamp(out4_2, -1, 1)
        out4_2 += residual3
        out4_2 = FakeQuantize.apply(out4_2, 128)
        out4_2 = torch.clamp(out4_2, -1, 1)
        out4_2 = self.relu(out4_2)
        
        out = self.avg_pool(out4_2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 定义ResNet9 Quantiz模型
class ResNet9_Quantiz(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet9_Quantiz, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_shortcut1 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0, bias=True)

        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_shortcut2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=True)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def quantize(self, scale_factors):
        self.conv1.weight.data = self.conv1.weight.data.mul(scale_factors['conv1.weight']).round().clamp(-128, 127)
        self.conv1.bias.data = self.conv1.bias.data.mul(scale_factors['conv1.bias']).round()
        self.conv2_1.weight.data = self.conv2_1.weight.data.mul(scale_factors['conv2_1.weight']).round().clamp(-128, 127)
        self.conv2_1.bias.data = self.conv2_1.bias.data.mul(scale_factors['conv2_1.bias']).round()
        self.conv2_2.weight.data = self.conv2_2.weight.data.mul(scale_factors['conv2_2.weight']).round().clamp(-128, 127)
        self.conv2_2.bias.data = self.conv2_2.bias.data.mul(scale_factors['conv2_2.bias']).round()
        self.conv3_1.weight.data = self.conv3_1.weight.data.mul(scale_factors['conv3_1.weight']).round().clamp(-128, 127)
        self.conv3_1.bias.data = self.conv3_1.bias.data.mul(scale_factors['conv3_1.bias']).round()
        self.conv3_2.weight.data = self.conv3_2.weight.data.mul(scale_factors['conv3_2.weight']).round().clamp(-128, 127)
        self.conv3_2.bias.data = self.conv3_2.bias.data.mul(scale_factors['conv3_2.bias']).round()
        self.conv_shortcut1.weight.data = self.conv_shortcut1.weight.data.mul(scale_factors['conv_shortcut1.weight']).round().clamp(-128, 127)
        self.conv_shortcut1.bias.data = self.conv_shortcut1.bias.data.mul(scale_factors['conv_shortcut1.bias']).round()
        self.conv4_1.weight.data = self.conv4_1.weight.data.mul(scale_factors['conv4_1.weight']).round().clamp(-128, 127)
        self.conv4_1.bias.data = self.conv4_1.bias.data.mul(scale_factors['conv4_1.bias']).round()
        self.conv4_2.weight.data = self.conv4_2.weight.data.mul(scale_factors['conv4_2.weight']).round().clamp(-128, 127)
        self.conv4_2.bias.data = self.conv4_2.bias.data.mul(scale_factors['conv4_2.bias']).round()
        self.conv_shortcut2.weight.data = self.conv_shortcut2.weight.data.mul(scale_factors['conv_shortcut2.weight']).round().clamp(-128, 127)
        self.conv_shortcut2.bias.data = self.conv_shortcut2.bias.data.mul(scale_factors['conv_shortcut2.bias']).round()
        self.fc.weight.data = self.fc.weight.data.mul(scale_factors['fc.weight']).round().clamp(-128, 127)
        self.fc.bias.data = self.fc.bias.data.mul(scale_factors['fc.bias']).round()

    def forward(self, x, scale_factors):
        x = self.conv1(x)
        x = cut(x, in_cut_start=int(log(scale_factors['conv1.weight'], 2)), en=True, bit_width=8)
        x = self.relu(x)

        residual1 = x
        out2_1 = self.conv2_1(x)
        out2_1 = cut(out2_1, in_cut_start=int(log(scale_factors['conv2_1.weight'], 2)), en=True, bit_width=8)
        out2_1 = self.relu2_1(out2_1)
        out2_2 = self.conv2_2(out2_1)
        out2_2 = cut(out2_2, in_cut_start=int(log(scale_factors['conv2_2.weight'], 2)), en=True, bit_width=8)
        out2_2 += residual1
        out2_2 = cut(out2_2, in_cut_start=int(log(scale_factors['conv2_2.weight'], 2)), en=True, bit_width=8)
        out2_2 = self.relu(out2_2)

        residual2 = self.conv_shortcut1(out2_2)
        residual2 = cut(residual2, in_cut_start=int(log(scale_factors['conv_shortcut1.weight'], 2)), en=True, bit_width=8)
        out3_1 = self.conv3_1(out2_2)
        out3_1 = cut(out3_1, in_cut_start=int(log(scale_factors['conv3_1.weight'], 2)), en=True, bit_width=8)
        out3_1 = self.relu3_1(out3_1)
        out3_2 = self.conv3_2(out3_1)
        out3_2 = cut(out3_2, in_cut_start=int(log(scale_factors['conv3_2.weight'], 2)), en=True, bit_width=8)
        out3_2 += residual2
        out3_2 = cut(out3_2, in_cut_start=int(log(scale_factors['conv3_2.weight'], 2)), en=True, bit_width=8)
        out3_2 = self.relu(out3_2)

        residual3 = self.conv_shortcut2(out3_2)
        residual3 = cut(residual3, in_cut_start=int(log(scale_factors['conv_shortcut2.weight'], 2)), en=True, bit_width=8)
        out4_1 = self.conv4_1(out3_2)
        out4_1 = cut(out4_1, in_cut_start=int(log(scale_factors['conv4_1.weight'], 2)), en=True, bit_width=8)
        out4_1 = self.relu4_1(out4_1)
        out4_2 = self.conv4_2(out4_1)
        out4_2 = cut(out4_2, in_cut_start=int(log(scale_factors['conv4_2.weight'], 2)), en=True, bit_width=8)
        out4_2 += residual3
        out4_2 = cut(out4_2, in_cut_start=int(log(scale_factors['conv4_2.weight'], 2)), en=True, bit_width=8)
        out4_2 = self.relu(out4_2)
        
        out = self.avg_pool(out4_2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = cut(out, in_cut_start=int(log(scale_factors['fc.weight'], 2)), en=True, bit_width=8)
        return out


# # 打印权重和偏置的最大值最小值
# def print_weights_and_biases():
#     parameters = {
#         "conv1.weight": net.conv1.weight,
#         "conv1.bias": net.conv1.bias,
#         "conv2_1.weight": net.conv2_1.weight,
#         "conv2_1.bias": net.conv2_1.bias,
#         "conv2_2.weight": net.conv2_2.weight,
#         "conv2_2.bias": net.conv2_2.bias,
#         "conv3_1.weight": net.conv3_1.weight,
#         "conv3_1.bias": net.conv3_1.bias,
#         "conv3_2.weight": net.conv3_2.weight,
#         "conv3_2.bias": net.conv3_2.bias,
#         "conv_shortcut1.weight": net.conv_shortcut1.weight,
#         "conv_shortcut1.bias": net.conv_shortcut1.bias,
#         "conv4_1.weight": net.conv4_1.weight,
#         "conv4_1.bias": net.conv4_1.bias,
#         "conv4_2.weight": net.conv4_2.weight,
#         "conv4_2.bias": net.conv4_2.bias,
#         "conv_shortcut2.weight": net.conv_shortcut2.weight,
#         "conv_shortcut2.bias": net.conv_shortcut2.bias,
#         "fc.weight": net.fc.weight,
#         "fc.bias": net.fc.bias,
#     }

#     scale_factors = {}
#     for param_name, param in parameters.items():
#         if param is not None:
#             max_val = param.max().item()
#             min_val = param.min().item()
#             max_abs_value = max(abs(max_val), abs(min_val))
#             scale_factor = nearest_power_of_2(calculate_scale_factor(max_abs_value))
#             scale_factors[param_name] = scale_factor
#             logger.info(f'{param_name} - max abs value: {max_abs_value}, scale factor: {scale_factor:.4f}, nearest power of 2: {scale_factor}')

#     return scale_factors

# 训练和验证模型
def train_and_validate():
    
    # CIFAR-10数据集加载和预处理
    transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    trainset = torchvision.datasets.CIFAR10(root='/data/CIFAR10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='/data/CIFAR10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 定义模型、损失函数和优化器
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net = ResNet9().to(device)
    net_quant_aware = ResNet9_QuantizAware().to(device)
    net_quant = ResNet9_Quantiz().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-08)
    
    epochs = 100
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 梯度置0
            optimizer.zero_grad()

            # 训练ResNet9
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f'Epoch [{epoch+1}], Step [{batch_idx+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

        # 验证ResNet9
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f'ResNet9 Accuracy after Epoch {epoch+1}: {100 * correct / total:.2f}%')

    # # 获取缩放因子
    # scale_factors = print_weights_and_biases()

    # # 加载ResNet9的权重和偏置到ResNet9_QuantizAware中
    # net_quant_aware.load_state_dict(net.state_dict())
    # net_quant_aware.add_quantizer(scale_factors)

    # for epoch in range(epochs):
    #     net_quant_aware.train()
    #     running_loss = 0.0

    #     for batch_idx, (inputs, targets) in enumerate(trainloader):
    #         inputs, targets = inputs.to(device), targets.to(device)

    #         # 梯度置0
    #         optimizer.zero_grad()

    #         # 量化感知训练
    #         outputs = net_quant_aware(inputs)  # 使用量化感知模型
    #         loss = criterion(outputs, targets)

    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()
    #         if batch_idx % 100 == 99:
    #             print(f'Epoch [{epoch+1}], Step [{batch_idx+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}')
    #             running_loss = 0.0

    #     # 验证量化感知模型
    #     net_quant_aware.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for inputs, targets in testloader:
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             outputs = net_quant_aware(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += targets.size(0)
    #             correct += (predicted == targets).sum().item()
    #     print(f'QuantizAware Accuracy after Epoch {epoch+1}: {100 * correct / total:.2f}%')

    #     # 验证量化模型
    #     net_quant_aware.remove_quantizer()
    #     net_quant.load_state_dict(net_quant_aware.state_dict())  # 加载训练好的参数
    #     net_quant.quantize(scale_factors)

    #     net_quant.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for inputs, targets in testloader:
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             quantized_inputs = inputs.mul((2**(8-1))).round().clamp(-128, 127)
    #             outputs = net_quant(quantized_inputs, scale_factors)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += targets.size(0)
    #             correct += (predicted == targets).sum().item()
    #     print(f'Quantized Model Accuracy after Epoch {epoch+1}: {100 * correct / total:.2f}%')

    #     net_quant_aware.add_quantizer(scale_factors)


if __name__ == '__main__':    
    train_and_validate()
