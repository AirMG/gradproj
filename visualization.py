
import torch
from net import FastSal
from torchviz import make_dot

img = torch.randn(1, 3, 256, 256)
depth = torch.randn(1, 1, 256, 256)
# x=torch.rand(8,3,256,512)
model=FastSal()
# model.eval()
# torch.cuda.synchronize()
y=model(img,depth)
# generate network picture
g = make_dot(y)
g.render('espnet_model', view=False)