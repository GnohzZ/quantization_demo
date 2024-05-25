import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import ppq.lib as PFL
from ppq import TargetPlatform, TorchExecutor, graphwise_error_analyse
from ppq.api import ENABLE_CUDA_KERNEL
from ppq.api.interface import load_onnx_graph
from ppq.core.quant import (QuantizationPolicy, QuantizationProperty,
                            RoundingPolicy)
from ppq.quantization.optim import (LearnedStepSizePass, ParameterBakingPass,
                                    ParameterQuantizePass, QuantAlignmentPass,
                                    QuantizeFusionPass, QuantizeSimplifyPass,
                                    RuntimeCalibrationPass)


graph = load_onnx_graph(onnx_import_file='./ckpt/resnet9.onnx')
test_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])
test_dataset = torchvision.datasets.CIFAR10('/data/CIFAR10', train=False, download=False, transform=test_transform)
dataset = [test_dataset[i][0].unsqueeze(0) for i in range(100)]

collate_fn  = lambda x: x.cuda()

from ppq import TensorQuantizationConfig as TQC

MyTQC = TQC(
    policy = QuantizationPolicy(
        QuantizationProperty.SYMMETRICAL + 
        QuantizationProperty.LINEAR +
        QuantizationProperty.PER_TENSOR +
        QuantizationProperty.POWER_OF_2),
    rounding=RoundingPolicy.ROUND_HALF_EVEN,
    num_of_bits=8, quant_min=-128, quant_max=127, 
    exponent_bits=0, channel_axis=None,
    observer_algorithm='minmax'
)

quantizer = PFL.Quantizer(platform=TargetPlatform.FPGA_INT8, graph=graph)

dispatching = PFL.Dispatcher(graph=graph).dispatch(                       
    quant_types=quantizer.quant_operation_types)

for op in graph.operations.values():
    dispatching['/conv1/Conv'] = TargetPlatform.FPGA_INT8
    dispatching['/relu1/Relu'] = TargetPlatform.FPGA_INT8
    dispatching['/fc/Gemm'] = TargetPlatform.FPGA_INT8
    quantizer.quantize_operation(
        op_name = op.name, platform = dispatching[op.name])

executor = TorchExecutor(graph=graph, device='cuda')
executor.tracing_operation_meta(inputs=collate_fn(dataset[0]))
executor.load_graph(graph=graph)

pipeline = PFL.Pipeline([
    QuantizeSimplifyPass(),
    QuantizeFusionPass(
        activation_type=quantizer.activation_fusion_types),
    ParameterQuantizePass(),
    RuntimeCalibrationPass(),
    QuantAlignmentPass(force_overlap=True),
    LearnedStepSizePass(
         steps=1000, is_scale_trainable=True, 
        lr=1e-5, block_size=4, collecting_device='cuda'),
    ParameterBakingPass()
])

# 调用管线完成量化
pipeline.optimize(
    graph=graph, dataloader=dataset, verbose=True, 
    calib_steps=32, collate_fn=collate_fn, executor=executor)

# 执行量化误差分析
graphwise_error_analyse(
    graph=graph, running_device='cuda', 
    dataloader=dataset, collate_fn=collate_fn)


exporter = PFL.Exporter(platform=TargetPlatform.ONNXRUNTIME)
exporter.export(file_path='./ckpt/Quantized.onnx', graph=graph)