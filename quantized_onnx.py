import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms



def onnx_modify():
    import onnx
    from onnx import helper, TensorProto
    model = onnx.load("./ckpt/Quantized.onnx")
    
    for i, node in enumerate(model.graph.node):
        # print(node.op_type)
        if node.op_type == "Reshape":
            for attr in node.attribute:
                if attr.name == "allowzero":
                    node.attribute.remove(attr)

    onnx.save(model, "./ckpt/MQuantized.onnx")


def onnx_infer():

    import onnxruntime as ort

    session = ort.InferenceSession("./ckpt/MQuantized.onnx", providers=["CPUExecutionProvider"])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                ])

    test_dataset = torchvision.datasets.CIFAR10('/data/CIFAR10', train=False, download=False, transform=test_transform)
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    outputs = session.run(None, {session.get_inputs()[0].name: test_dataset[0][0].unsqueeze(0).numpy()})

    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            images, labels = data
            images = images.numpy()

            outputs = session.run(None, {session.get_inputs()[0].name: images})[0]
            _, predicted = torch.max(torch.Tensor(outputs), dim=1)
            total += labels.size(0)
            correct += (predicted == labels.flatten()).sum().item()

    return correct / total
    
    
if __name__ == "__main__":
    # onnx_modify()
    print(onnx_infer())