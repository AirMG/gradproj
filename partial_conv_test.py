import torch
import torch.nn as nn
import time

import torch
import torch.nn as nn
import time
size=512
in_channel=64
out_channel=32
batch=1
inp = torch.randn((batch, in_channel, size, size)).cuda()
w = torch.randn((out_channel, in_channel, 3, 3)).cuda()
m = nn.ConstantPad2d(1, 0).cuda()
inp_padding = m(inp)

inp2 = inp
w2 = w.clone()
inp_padding2 = inp_padding

torch.cuda.synchronize()
t1=time.time()
# print(inp)
# mask = torch.Tensor([[[[1,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]]])
# mask = (inp>0)[0:1,0:1,...].type(torch.float64)
mask = torch.ones((1,1,size,size)).cuda()
# mask = torch.zeros((1,1,size,size)).cuda()

mask = m(mask)

torch.cuda.synchronize()
t2=time.time()

inp_unf = torch.nn.functional.unfold(inp_padding, (3,3))
# print(inp_unf)
mask_unf = torch.nn.functional.unfold(mask, (3,3))

torch.cuda.synchronize()
t3=time.time()

# print(mask_unf.max(dim=1,keepdim=False)[0].sum())
index = mask_unf.max(dim=1,keepdim=True)[0].squeeze().nonzero().squeeze()

torch.cuda.synchronize()
t4=time.time()

inp_unf_slct = inp_unf.index_select(2,index)

torch.cuda.synchronize()
t5=time.time()

# print(w.view(w.size(0), -1).shape)
wt = w.view(w.size(0), -1)
# print(wt.shape, inp_unf_slct.shape)

torch.cuda.synchronize()
t6=time.time()

out_unf = wt.matmul(inp_unf_slct)
# out_unf = wt.matmul(torch.zeros(0).cuda())


torch.cuda.synchronize()
t7=time.time()
print(t7-t6)
# for k in range(len(index)):
#     inp[...,index[k]//1024,index[k]-1024*(index[k]//1024)]=out_unf[k]

out1 = inp.view(inp.shape[0],inp.shape[1],-1).permute(2,0,1)
print(out1[index,...].shape, out_unf.permute(2,0,1).shape)
out1 = out1[...,0:out_channel]
out1[index,...] = out_unf.permute(2,0,1)

out1 = out1.permute(1,2,0).view(inp.shape[0],out_channel,inp.shape[2],inp.shape[3])

torch.cuda.synchronize()
t8=time.time()

torch.cuda.synchronize()
t10=time.time()
# conv = nn.Conv2d(1,1,3,1,1,bias=False).cuda()
# for i in range(50):
torch.cuda.synchronize()
t11=time.time()
# conv(inp2)
out2=torch.nn.functional.conv2d(inp_padding2, w2)

torch.cuda.synchronize()
t12=time.time()




print(t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6,t8-t7,t8-t1,t11-t10,t12-t11)
print((out1-out2).abs().max())