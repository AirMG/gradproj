import torch
import torch.nn as nn
import time

class partial_conv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3,padding=1,bias=True):
        super(partial_conv, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channel,in_channel,kernel,kernel))
        # self.weight = torch.randn((out_channel,in_channel,kernel,kernel)).cuda()
        # print(self.weight)
        self.padding = nn.ConstantPad2d(1, 0)
        self.out_channel = out_channel
        if bias == True:
            self.bias = nn.Parameter(torch.randn((out_channel,1)))
    def forward(self,x,m):
        b,c,h,w = x.shape
        x_padding = self.padding(x)
        m_padding = self.padding(m)
        inp_unfold = torch.nn.functional.unfold(x_padding, (3, 3))
        mask_unfold = torch.nn.functional.unfold(m_padding, (3, 3))
        index = mask_unfold.max(dim=1,keepdim=True)[0].squeeze().nonzero().squeeze()
        inp_unfold_slct = inp_unfold.index_select(2,index)
        # print(wt.shape,inp_unfold_slct.shape)
        mm = self.weight.view(self.weight.size(0), -1).matmul(inp_unfold_slct) + self.bias
        out = x.view(b,c,-1).permute(2,0,1)  # not in-place
        out = out[..., 0:self.out_channel]
        out = out.clone()
        out[index,...] = mm.permute(2,0,1)
        out = out.permute(1,2,0).view(b,self.out_channel,h,w)

        return out

if __name__ == '__main__':
    conv1 = partial_conv(64,64).cuda()
    conv2 = nn.Conv2d(64,64,3,1,1,bias=False).cuda()
    inp = torch.randn((1,64,128,128)).cuda()
    print(inp)
    torch.cuda.synchronize()
    t1=time.time()
    print(conv1(inp,torch.zeros(1,1,128,128).cuda()))
    torch.cuda.synchronize()
    t2=time.time()
    print(conv2(inp))
    torch.cuda.synchronize()
    t3=time.time()
    print(t3-t2,t2-t1)