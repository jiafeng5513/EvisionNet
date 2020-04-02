import torch

image = torch.randn(1, 4, 128, 416)  # 输入尺寸
image2 = torch.randn(1, 4, 128, 416)  # 输入尺寸

out = torch.cat([image, image2], dim=-1)
# N, C, H, W = layer.shape
print(out.shape)

