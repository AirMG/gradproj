import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mobilenet import MobileNetV2Encoder
import time
from heatmap import  heatmap

INPUT_SIZE = 512

a = torch.randn(3,4)
print(a.size(0))
# print(MobileNetV2Encoder(1).layer1)
# print(MobileNetV2Encoder(3))