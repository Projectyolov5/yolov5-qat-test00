import torch

import copy

from yolo_easy import yolov5s6, QuantizedNet


cpu = torch.device("cpu:0")
gpu = torch.device("cuda:0")

model = yolov5s6()
fused_model = copy.deepcopy(model)

def dfs_fuse(module):
    for m_name, m in module.named_children():
        print(m_name)
        if "Conv" in m_name:
            torch.quantization.fuse_modules(m, [["conv", "bn", "act"]], inplace=True)
            continue
        if "cv" in m_name:
            torch.quantization.fuse_modules(m, [["conv", "bn", "act"]], inplace=True)
            continue
        dfs_fuse(m)

dfs_fuse(fused_model)

# for module_name, module in fused_model.named_children():
#     print(module_name)
# for module_name, module in fused_model.named_children():
#     if "Conv" in module_name:
#         for basic_block_name, basic_block in module.named_children():
#             torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
#             for sub_block_name, sub_block in basic_block.named_children():
#                 if sub_block_name == "downsample":
#                     torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
        
#     if "C3" in module_name:
#         for basic_block_name, basic_block in module.named_children():
#             torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
#             for sub_block_name, sub_block in basic_block.named_children():
#                 if sub_block_name == "downsample":
#                     torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

# # Print FP32 model.
# print(model)
# # Print fused model.
# print(fused_model)