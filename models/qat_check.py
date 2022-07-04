# import torch

# from copy import copy

from yolo_easy import *
from common2 import *
from quantization_aware_training import *
import copy

cpu = torch.device("cpu:0")
gpu = torch.device("cuda:0")

model = yolov5s6()
fused_model = copy.deepcopy(model)

Output = output_maker()

def dfs_fuse(module):
    for m_name, m in module.named_children():
        if isinstance(m, Conv):
            torch.quantization.fuse_modules(m, [["conv", "bn", "act"]], inplace=True)
            continue
        dfs_fuse(m)

dfs_fuse(fused_model)



# # Print FP32 model.
# print(model)
# # Print fused model.
# print(fused_model)

# Model and fused model should be equivalent.
model.eval()
fused_model.eval()
# assert model_equivalence(model_1=model, model_2=fused_model, device=cpu, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,1280,1280)), "Fused model is not equivalent to the original model!"

# Prepare the model for quantization aware training. This inserts observers in
# the model that will observe activation tensors during calibration.
quantized_model = QuantizedNet(model_fp32=fused_model)
# Using un-fused model will fail.
# Because there is no quantized layer implementation for a single batch normalization layer.
# quantized_model = QuantizedResNet18(model_fp32=model)
# Select quantization schemes from 
# https://pytorch.org/docs/stable/quantization-support.html
quantization_config = torch.quantization.get_default_qat_qconfig("fbgemm")
# Custom quantization configurations
# quantization_config = torch.quantization.default_qconfig
# quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

quantized_model.qconfig = quantization_config
    
# Print quantization configurations
# print(quantized_model.qconfig)
quantized_model = torch.quantization.prepare_qat(quantized_model, inplace=True)

quantized_model = torch.quantization.convert(quantized_model, inplace=True)

# print(quantized_model)
def check_quantized(module):
    for m_name, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            print(m)
            exit()
        check_quantized(m)

check_quantized(quantized_model)
# print(quantized_model)

model_dir = "./"
quantized_model_filename = "yolov5s6-qat.pt"
quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)


# Save quantized model.
save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)

# Load quantized model.
quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath, device=cpu)
# print(quantized_jit_model)


fp32_cpu_inference_latency = measure_inference_latency(model=[model, Output], device=cpu, input_size=(1,3,1280,1280), num_samples=100)
int8_cpu_inference_latency = measure_inference_latency(model=[quantized_model, Output], device=cpu, input_size=(1,3,1280,1280), num_samples=100)
int8_jit_cpu_inference_latency = measure_inference_latency(model=[quantized_jit_model, Output], device=cpu, input_size=(1,3,1280,1280), num_samples=100)
fp32_gpu_inference_latency = measure_inference_latency(model=[model, Output], device=gpu, input_size=(1,3,1280,1280), num_samples=100)
    
print("FP32 CPU Inference Latency: {:.2f} ms / sample,".format(fp32_cpu_inference_latency * 1000), "FPS: %f" % (1/fp32_cpu_inference_latency))
print("FP32 CUDA Inference Latency: {:.2f} ms / sample,".format(fp32_gpu_inference_latency * 1000), "FPS: %f" % (1/fp32_gpu_inference_latency))
print("INT8 CPU Inference Latency: {:.2f} ms / sample,".format(int8_cpu_inference_latency * 1000), "FPS: %f" % (1/int8_cpu_inference_latency))
print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample,".format(int8_jit_cpu_inference_latency * 1000), "FPS: %f" % (1/int8_jit_cpu_inference_latency))
