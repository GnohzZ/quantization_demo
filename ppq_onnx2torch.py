import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


onnx_mode_path = "./ckpt/Quantized.onnx"


def findInitializers(model: onnx.ModelProto, data_name: str):
    def onnx_datatype_to_npType(data_type):
        if data_type == 1:
            return np.float32
        else:
            raise TypeError("don't support data type")

    for initializer in model.graph.initializer:
        if initializer.name == data_name:
            dtype = initializer.data_type
            if len(initializer.dims) == 0:
                if dtype == 1:
                    params = initializer.float_data[0]
                elif dtype == 3:
                    params = initializer.int32_data[0]
            else:
                params = np.frombuffer(initializer.raw_data, dtype=onnx.helper.tensor_dtype_to_np_dtype(dtype)).reshape(initializer.dims)
                params = np.copy(params)
            return params
    raise ValueError(f"Can't find {data_name} in initializers")


def findNode(model: onnx.ModelProto, node_name: str):
    for i, node in enumerate(model.graph.node):
        if node.name == node_name:
            return node
    raise ValueError(f"Can't find {node_name} in nodes")


def findNodeWithOutput(model: onnx.ModelProto, output_name: str):
    for i, node in enumerate(model.graph.node):
        for output in node.output:
            if output == output_name:
                return node
    raise ValueError(f"Can't find node with output {output_name} in nodes")


def findNodeWithInput(model: onnx.ModelProto, input_name: str):
    for i, node in enumerate(model.graph.node):
        for input in node.input:
            if input == input_name:
                return node
    raise ValueError(f"Can't find node with input {input_name} in nodes")


def getConvInfo(model: onnx.ModelProto, node_name: str):
    info = {}
    for i, node in enumerate(model.graph.node):
        if node.name == node_name:
            # op_type
            assert node.op_type == "Conv"
            info["op_type"] = node.op_type
            
            # weight
            weight_dq_name = node.input[1]
            weight_dq = findNodeWithOutput(model, weight_dq_name)
            assert weight_dq.op_type == "DequantizeLinear"
            info["weight_scale"] = findInitializers(model, weight_dq.input[1])
            info["weight_zero_point"] = findInitializers(model, weight_dq.input[2])
            
            weight_q_name = weight_dq.input[0]
            weight_q = findNodeWithOutput(model, weight_q_name)
            assert weight_q.op_type == "QuantizeLinear"
            info["weight"] = findInitializers(model, weight_q.input[0])
            
            # input
            input_dq_name = node.input[0]
            input_dq = findNodeWithOutput(model, input_dq_name)
            while input_dq.op_type != "DequantizeLinear":
                input_dq = findNodeWithOutput(model, input_dq.input[0])
            info["input_scale"] = findInitializers(model, input_dq.input[1])
            info["input_zero_point"] = findInitializers(model, input_dq.input[2])
            
            # output
            output_q_name = node.output[0]
            output_q = findNodeWithInput(model, output_q_name)
            while output_q.op_type != "QuantizeLinear":
                output_q = findNodeWithInput(model, output_q.output[0])
            info["output_scale"] = findInitializers(model, output_q.input[1])
            info["output_zero_point"] = findInitializers(model, output_q.input[2])
            
             # bias
            bias_name = node.input[2]
            info["bias"] = findInitializers(model, bias_name)
            info["bias_scale"] = info["input_scale"] * info["weight_scale"]
            info["bias_zero_point"] = 0    
            
            return info       
            
    raise ValueError(f"Can't find {node_name} in nodes")


def getFcInfo(model: onnx.ModelProto, node_name: str):
    info = {}
    
    for i, node in enumerate(model.graph.node):
        if node.name == node_name:
            # op_type
            assert node.op_type == "Gemm"
            info["op_type"] = node.op_type
            
            # weight
            weight_dq_name = node.input[1]
            weight_dq = findNodeWithOutput(model, weight_dq_name)
            assert weight_dq.op_type == "DequantizeLinear"
            info["weight_scale"] = findInitializers(model, weight_dq.input[1])
            info["weight_zero_point"] = findInitializers(model, weight_dq.input[2])
            
            weight_q_name = weight_dq.input[0]
            weight_q = findNodeWithOutput(model, weight_q_name)
            assert weight_q.op_type == "QuantizeLinear"
            info["weight"] = findInitializers(model, weight_q.input[0])
            
            # input
            input_dq_name = node.input[0]
            input_dq = findNodeWithOutput(model, input_dq_name)
            while input_dq.op_type != "DequantizeLinear":
                input_dq = findNodeWithOutput(model, input_dq.input[0])
            info["input_scale"] = findInitializers(model, input_dq.input[1])
            info["input_zero_point"] = findInitializers(model, input_dq.input[2])
            
            # output
            output_q_name = node.output[0]
            output_q = findNodeWithInput(model, output_q_name)
            while output_q.op_type != "QuantizeLinear":
                output_q = findNodeWithInput(model, output_q.output[0])
            info["output_scale"] = findInitializers(model, output_q.input[1])
            info["output_zero_point"] = findInitializers(model, output_q.input[2])
            
             # bias
            bias_name = node.input[2]
            info["bias"] = findInitializers(model, bias_name)
            info["bias_scale"] = info["input_scale"] * info["weight_scale"]
            info["bias_zero_point"] = 0
            
            return info
    
    raise ValueError(f"Can't find {node_name} in nodes")


def getAddInfo(model: onnx.ModelProto, node_name: str):
    info = {}
    
    for i, node in enumerate(model.graph.node):
        if node.name == node_name:
            # op_type
            assert node.op_type == "Add"
            info["op_type"] = node.op_type
            
            # input 1
            input_dq_name = node.input[0]
            input_dq = findNodeWithOutput(model, input_dq_name)
            while input_dq.op_type != "DequantizeLinear":
                input_dq = findNodeWithOutput(model, input_dq.input[0])
            info["input_1_scale"] = findInitializers(model, input_dq.input[1])
            info["input_1_zero_point"] = findInitializers(model, input_dq.input[2])
            
            # input 2
            input_dq_name = node.input[1]
            input_dq = findNodeWithOutput(model, input_dq_name)
            while input_dq.op_type != "DequantizeLinear":
                input_dq = findNodeWithOutput(model, input_dq.input[0])
            info["input_2_scale"] = findInitializers(model, input_dq.input[1])
            info["input_2_zero_point"] = findInitializers(model, input_dq.input[2])
            
            # output
            output_q_name = node.output[0]
            output_q = findNodeWithInput(model, output_q_name)
            while output_q.op_type != "QuantizeLinear":
                output_q = findNodeWithInput(model, output_q.output[0])
            info["output_scale"] = findInitializers(model, output_q.input[1])
            info["output_zero_point"] = findInitializers(model, output_q.input[2])
            
            return info
    
    raise ValueError(f"Can't find {node_name} in nodes")


def extractInt8QuantizedOnnx(onnx_model_path):
    model = onnx.load(onnx_model_path)
    
    quantized_layer = {}
    
    conv_list = []
    fc_list = []
    add_list = []
    
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Conv":
            conv_list.append(node.name)
            quantized_layer[node.name] = {
                # "op_type": node.op_type,
                
                # "weight": None,
                # "weight_scale": None,
                # "weight_zero_point": None,
                
                # "bias": None,
                # "bias_scale": None,
                # "bias_zero_point": None,
                
                # "input_scale": None,
                # "input_zero_point": None,
                
                # "output_scale": None,
                # "output_zero_point": None,
            }
        elif node.op_type == "Gemm":
            fc_list.append(node.name)
            quantized_layer[node.name] = {
                # "op_type": node.op_type,
                
                # "weight": None,
                # "weight_scale": None,
                # "weight_zero_point": None,
                
                # "bias": None,
                # "bias_scale": None,
                # "bias_zero_point": None,
                
                # "input_scale": None,
                # "input_zero_point": None,
                
                # "output_scale": None,
                # "output_zero_point": None,
            }
        elif node.op_type == "Add":
            add_list.append(node.name)
            quantized_layer[node.name] = {
                # "op_type": node.op_type,
                
                # "input_1_scale": None,
                # "input_1_zero_point": None,
                
                # "input_2_scale": None,
                # "input_2_zero_point": None,
                
                # "output_scale": None,
                # "output_zero_point": None,
            }
            
    
    for _ in conv_list:
        quantized_layer[_] = getConvInfo(model, _)
    
    for _ in fc_list:
        quantized_layer[_] = getFcInfo(model, _)
    
    for _ in add_list:
        quantized_layer[_] = getAddInfo(model, _)
    
    return quantized_layer
        
        


if __name__ == "__main__":
    LayerInfo = extractInt8QuantizedOnnx(onnx_mode_path)
    print(len(LayerInfo))