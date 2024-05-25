import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from math import log
from custom_resnet9 import ResNet9
from quantized_resnet9 import ResNet9 as QuantizeAwareResnet9
from quantized_resnet9 import QuantizedResNet9
import util
import logging
from ppq_onnx2torch import extractInt8QuantizedOnnx

def train():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # net = resnet8().to(device)
    # net = Net().to(device)
    net = QuantizeAwareResnet9().to(device)
    net.add_quantizer()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
    net.train()

    train_transform = transforms.Compose([
        util.Cutout(num_cutouts=2, size=8, p=0.8),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='/data/CIFAR10', train=True, download=False, transform=train_transform)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    torch.autograd.set_detect_anomaly(True)
    
    num_epochs = 100
    best_accuracy = 0
    
    train_accuracies = []
    for epoch in range(0, num_epochs + 1):
        net.train()
        print('Epoch {}/{}'.format(epoch, num_epochs))

        epoch_correct = 0
        epoch_total = 0
        for i, data in enumerate(data_loader, 1):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net.forward(images)
            loss = criterion(outputs, labels.squeeze_())
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, dim=1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels.flatten()).sum().item()

            epoch_total += batch_total
            epoch_correct += batch_correct

        print("Loss: ", loss.item(), "Accuracy: ", epoch_correct / epoch_total)
        train_accuracies.append(epoch_correct / epoch_total)
        
        test_accuracy = test(net)
        print("test accuracy: ", test_accuracy)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(net.state_dict(), './ckpt/resnet9.pth')


def test(net):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    net.eval()
    test_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])

    test_dataset = torchvision.datasets.CIFAR10('/data/CIFAR10', train=False, download=False, transform=test_transform)
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.flatten()).sum().item()

    net.train()
    return correct / total


if __name__ == "__main__":
    # train()
    net = QuantizedResNet9(extractInt8QuantizedOnnx('./ckpt/Quantized.onnx'), 10)
    net.getParams()
    net.quantize()
    
    # net = QuantizeAwareResnet9()
    # net.getParams(extractInt8QuantizedOnnx('./ckpt/Quantized.onnx'))
    # net.add_quantizer()
    
    net = net.to('cuda:1')
    net.eval()
    print(test(net))
    
    