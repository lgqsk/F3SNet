import torch
from thop import profile
from thop import clever_format
from models.F3SNet import F3SNet as net


model = net(3, 32, False)
# model = net()
input = torch.randn(1, 3, 256, 256)
macs, params = profile(model, inputs=(input,input))
flops, params = clever_format([macs * 2, params], "%.3f")
print('params', params)
print('FLOPs', flops)