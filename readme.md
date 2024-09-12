# 使用PPQ量化ResNet9
本项目的文件没有进行整理，比较混乱，请参考此readme阅读。
## 量化流程
1. 将**预训练**的原模型导出为onnx
2. 使用PPQ进行PTQ（参考resnet9_ptq.py）
    * MyTQC应该不起作用
    * TargetPlatform.FPGA_INT8使用的Quantizer就是有2的整次幂约束的对称均匀Int8量化
    * 导出ONNXRUNTIME可以保留所有量化信息，其他导出平台在此项目测试环境下或多或少有问题
3. 可以直接用导出的ONNXRUNTIME进行推理，测试准确率（参考quantized_onnx.py）
4. 如果想QAT，可以尝试直接使用PPQ提供的QAT功能，模仿： https://github.com/openppl-public/ppq/blob/e39eecb9f7e5f017c28f180cb423f8a685c3db48/ppq/samples/QAT/myquantizer.py
    * PPQ不建议从头开始做QAT，一定是在PTQ的基础上做QAT!
5. 如果想将PPQ量化的结果放回torch，可以参考ppq_onnx2torch.py和quantized_resnet9.py中的QuantizedResNet9
    * 这里的逻辑是在onnx中提取量化参数，然后在torch中手动搭建量化网络并加载这些量化参数；这个流程比较复杂，容易出错
    * onnx转回torch是个深坑，尤其是加上量化的算子之后
    * ppq_onnx2torch.py写得比较通用，可以复用；如果网络更加复杂需要增加一些代码