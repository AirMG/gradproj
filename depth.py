import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import time
import timm
from heatmap import heatmap
from mobilenet import MobileNetV2Encoder



"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FastSal', 'get_fast_sal']


def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)

def initialize_weights(model):
    m = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    pretrained_dict = m.state_dict()
    all_params = {}
    for k, v in model.state_dict().items():
        if k in pretrained_dict.keys() :#and v.shape == pretrained_dict[k]:
            v = pretrained_dict[k]
            all_params[k] = v
    # assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
    model.load_state_dict(all_params)

class DepthBranch(nn.Module):
    def __init__(self, c1=8, c2=16, c3=32, c4=48, c5=320, **kwargs):
        super(DepthBranch, self).__init__()
        self.base = MobileNetV2Encoder(3)
        initialize_weights(self.base)
        self.kernel = torch.Tensor([[[[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]]]]).cuda()


        self.conv_s_d = nn.Sequential(_DSConv(c5, c5 // 4),
                                      _DSConv(c5 // 4, c5 // 16),
                                      nn.Conv2d(c5 // 16, 1, 1), )
        self.conv_e_d = RPP(c3, c2)
        self.dsconv_tran = _DSConv(c3, c3)
        self.focus = focus()
        # nn.Sequential(_DSConv(c3, c3 // 4),
        #                           nn.Conv2d(c3 // 4, 1, 1), )
        self.ca1 = nn.Sequential( nn.Conv2d(16, 16, 1, 1), nn.Sigmoid())
        self.ca2 = nn.Sequential( nn.Conv2d(24, 24, 1, 1), nn.Sigmoid())
        self.ca3 = nn.Sequential( nn.Conv2d(32, 32, 1, 1), nn.Sigmoid())
        self.ca3_1 = nn.Sequential( nn.Conv2d(32, 32, 1, 1), nn.Sigmoid())
        self.ca4 = nn.Sequential( nn.Conv2d(96, 96, 1, 1), nn.Sigmoid())
        self.ca5 = nn.Sequential( nn.Conv2d(320, 320, 1, 1), nn.Sigmoid())

    def forward(self, x):
        size = x.size()[2:]
        out = []
        feat = []
        # edge feature extraction

        # depth coarse sal
        x1,x2,x3,x4,x5 = self.base(x)
        x5 = self.ca5(F.adaptive_avg_pool2d( x5,1)) *  x5
        x4 = self.ca4(F.adaptive_avg_pool2d( x4,1)) *  x4
        x3 = self.ca3(F.adaptive_avg_pool2d( x3,1)) *  x3
        x2 = self.ca2(F.adaptive_avg_pool2d( x2,1)) *  x2
        x1 = self.ca1(F.adaptive_avg_pool2d( x1,1)) *  x1
        s_d = self.conv_s_d(x5)
        

        # use coarse sal to refine edge feature

        # predict depth edge


        # output
        out.append(s_d)
        out.append(s_d)
        feat.append(x1)
        feat.append(x2)
        feat.append(x3)
        feat.append(x4)
        feat.append(x5)
        return out ,feat

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

def _make_layer( block, inplanes, planes, blocks, t=6, stride=1):
    layers = []
    layers.append(block(inplanes, planes, t, stride))
    for i in range(1, blocks):
        layers.append(block(planes, planes, t, 1))
    return nn.Sequential(*layers)

class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    '''fast Decoder with gate of depth quality by overlap rate of bi-modal predicted edge '''

    def __init__(self):
        super(Decoder, self).__init__()
        self.k = 4
        self.mlp_match = nn.Sequential(nn.Linear(self.k,self.k*4),nn.ReLU(),nn.Linear(4*self.k,1),nn.Sigmoid())
        self.conv_cp = _DSConv(256,64)
        # self.conv_fusion = _DSConv(128, 64)
        self.score_final = nn.Sequential(_DSConv(64,64),nn.ReLU(), _DSConv(64,64),nn.ReLU(),nn.Conv2d(64,1,1))
    # def forward(self,fusion_high, feat3, feat2, feat1, feat1_depth, edge_depth, edge_rgb, sal_rgb, sal_depth):
    def forward(self, feature_fusion, sal_fusion_sig, feature_rgb, feature_depth, edge_rgb_sig, edge_depth_sig):
        b = feature_fusion.shape[0]
        feature_fusion = self.conv_cp(feature_fusion)
        feature_fusion = upsample(feature_fusion, feature_rgb.shape[2])
        acr_r = edge_rgb_sig.view(b,-1).sum(1) / sal_fusion_sig.view(b,-1).sum(1)
        acr_d = edge_depth_sig.view(b,-1).sum(1) / sal_fusion_sig.view(b,-1).sum(1)
        sum_edge_ratio = edge_rgb_sig.view(b,-1).sum(1)/edge_depth_sig.view(b,-1).sum(1)
        iou = 2 * (edge_rgb_sig * edge_depth_sig).view(b,-1).sum(1) / (edge_depth_sig.view(b,-1).sum(1)+ edge_depth_sig.view(b,-1).sum(1))
        sum_edge_ratio = sum_edge_ratio.unsqueeze(1)
        iou = iou.unsqueeze(1)
        acr_r = acr_r.unsqueeze(1)
        acr_d = acr_d.unsqueeze(1)
        gate = self.mlp_match(torch.cat((iou,sum_edge_ratio,acr_r,acr_d),dim=1))
        # sal_init = sal_rgb.sigmoid() + sal_depth.sigmoid()
        # print(feature_rgb.shape, feature_rgb.shape, gate.shape, feature_depth.shape)
        feature = feature_fusion + feature_rgb + gate.unsqueeze(1).unsqueeze(1)  * feature_depth
        print(gate[0].data.cpu())

        return self.score_final(feature),gate




class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out

class DepthEdge(nn.Module):
    ''' depth edge'''
    def __init__(self,channel=64):
        super(DepthEdge, self).__init__()
        self.kernel = torch.Tensor([[[[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]]]]).cuda()

        self.score_edge = nn.Sequential(_DSConv(channel, channel),nn.ReLU(),_DSConv(channel, channel),nn.ReLU(),nn.Conv2d(channel, 1, 1),)
    def forward(self, sal_coarse_sig,feat_d):
        size = feat_d.size()[2:]
        # sal_coarse_sig = F.conv2d(sal_coarse_sig, self.kernel,padding=1,dilation=5)
        sal_coarse_sig = upsample(sal_coarse_sig, size)
        # heatmap(feat)
        # heatmap(sal_coarse)
        feat = feat_d * sal_coarse_sig
        # heatmap(feat)
        edge_depth = self.score_edge(feat)
        # heatmap(edge_depth.sigmoid())
        return edge_depth #128*128*16
    
class FocusRGBEdge(nn.Module):
    '''focus on edge of RGB with the guide of edge depth'''
    def __init__(self,channel1=64,channel2=64,channel3=64):
        super(FocusRGBEdge,self).__init__()
        self.kernel = torch.Tensor([[[[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]]]]).cuda()
        self.channel =  channel2
        self.score_edge = nn.Sequential(_DSConv(self.channel, self.channel//2),nn.ReLU(),_DSConv(self.channel//2, self.channel//2),nn.ReLU(),nn.Conv2d(self.channel//2, 1, 1),)
    def forward(self, edge_depth_sig, feat_rgb):
        size = feat_rgb.size()[2:]
        edge_depth_sig = F.conv2d(edge_depth_sig, self.kernel, stride=1, padding=1, dilation=2)
        edge_depth_sig = upsample(edge_depth_sig, size)
        edge_rgb_focus = self.score_edge(feat_rgb * edge_depth_sig)
        return edge_rgb_focus

class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x

class Denoising(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels,rgb_feat_channel, **kwargs):
        super(Denoising, self).__init__()
        self.relu = nn.ReLU()
        self.pool1 = nn.AvgPool2d(4,4)
        self.pool2 = nn.AvgPool2d(8,8)
        self.pool3 = nn.AvgPool2d(16,16)
        self.spatial_att = _DSConv(rgb_feat_channel,1)
        self.out = _DSConv(in_channels, out_channels, 2)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def forward(self, x, r):
        size = x.size()[2:0]
        feat1 = upsample(self.pool1(x), size)
        feat2 = upsample(self.pool2(x), size)
        feat3 = upsample(self.pool3(x), size)
        feat4 =  x
        # heatmap(rgb_tran)
        # heatmap(feat4)
        x = feat1 + feat2 + feat3 + feat4
        # heatmap(x)
        return self.out(x),feat1,feat2,feat3,feat4


class RGBBranch(nn.Module):
    """RGBBranch for low-level RGB feature extract"""

    def __init__(self, c1=32, c2=64, c3=64, c4=96,c5=128,k=32 ,**kwargs):
        super(RGBBranch, self).__init__()
        self.conv = _ConvBNReLU(3, c1, 3, 2, 1)
        self.dsconv1 = _DSConv(c1, c2, 2)
        self.bottleneck1 = _make_layer(block=LinearBottleneck, inplanes=c2, planes=c3, blocks=3, t=6, stride=2)
        self.bottleneck2 = _make_layer(LinearBottleneck, c3, c4, blocks=3, t=6, stride=2)
        self.bottleneck3 = _make_layer(LinearBottleneck, c4, c5, blocks=3, t=6, stride=2)
        self.ppm = PyramidPooling(c5, c5)
        self.conv_s_f = nn.Sequential(_DSConv(c5, c5 // 4),
                                      _DSConv(c5 // 4, c5 // 16),
                                      nn.Conv2d(c5 // 16, 1, 1), )
        self.conv_e_r = nn.Sequential(_DSConv(c3, c3 // 4),
                                      nn.Conv2d(c3 // 4, 1, 1), )
        self.conv_tran = _DSConv(c2,c2,1)
        self.conv_cp1 = _DSConv(c2,k)
        self.conv_cp2 = _DSConv(c3, k)
        self.conv_cp3 = _DSConv(c4, k)
        self.conv_cp3_1 = _DSConv(k, k)
        self.conv_cp4 = _DSConv(c5, k)
        self.conv_cp5 = _DSConv(c5, k)
        self.conv_s_f = nn.Sequential(_DSConv(3 * k,  k),
                                          _DSConv( k, k),
                                          nn.Conv2d(k, 1, 1), )
        self.HA = Refine()
        self.agg1 = aggregation_init(32)
        self.agg2 = aggregation_final(32)

    def forward(self, x, e_d, f_d):
        x = self.conv(x)
        x = self.dsconv1(x)

        
        f_l = x
        
        e_d_sig = upsample(e_d, x.shape[2:]).sigmoid()
        e_d_sig = F.avg_pool2d(e_d_sig, 3, 2, 1)
        e_d_sig = upsample(e_d_sig, f_l.shape[2:])
        f_l_r = e_d_sig * f_l
        e_r = self.conv_e_r(f_l_r)
        att = e_r.sigmoid() * e_d_sig
        att = F.avg_pool2d(att, 4, 4)
        att = upsample(att, f_l.shape[2:])
        
        x = att * upsample(self.conv_tran(f_d), x.shape[2:]) + x
        r1 = x
        
        x = self.bottleneck1(x)
        r2 = x
        x = self.bottleneck2(x)
        r3 = x
        x = self.bottleneck3(x)
        r4 = x

        x = self.ppm(x)
        r5 = x

        r5 = self.conv_cp5(r5)
        r4 = self.conv_cp4(r4)
        r3 = self.conv_cp3(r3)
        s_c = self.agg1(r5, r4, r3)

        # Refine low-layer features by initial map
        r1, r2, r3 = self.HA(s_c.sigmoid(), r1, r2, r3)

        # produce final saliency map by decoder2
        r3 = self.conv_cp3_1(r3)
        r2 = self.conv_cp2(r2)
        r1 = self.conv_cp1(r1)
        y = self.agg2(r3,r2, r1)  # *4

        s_f = self.conv_s_f (y)
        
        return s_f,s_c,e_r

class Depth(nn.Module):
    """RGBBranch for low-level Depth feature extract"""

    def __init__(self, c1=8,c2=16,c3=32,c4=48,c5=64, **kwargs):
        super(Depth, self).__init__()
        self.kernel = torch.Tensor([[[[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]]]]).cuda()
        self.conv_d_l1 = _ConvBNReLU(1,c1,3,2,1)
        self.conv_d_l2 = _DSConv(c1, c3, 2)
        self.conv = _ConvBNReLU(1, c1, 3, 2,1)
        self.dsconv0 = _DSConv(c1, c2, 2)
        self.dsconv1 = _DSConv(c2, c3,2)
        self.dsconv2 = _DSConv(c3, c4,2)
        self.bottleneck = _make_layer(LinearBottleneck, c4, c5, blocks=3, t=6, stride=2)
        self.conv_s_d = nn.Sequential(_DSConv(c5, c5//4),
                                        _DSConv(c5//4, c5//16),
                                        nn.Conv2d(c5//16, 1, 1), )
        self.conv_e_d = RPP(c3,c2)
        self.dsconv_tran = _DSConv(c3,c3)
        self.focus = focus()
            # nn.Sequential(_DSConv(c3, c3 // 4),
            #                           nn.Conv2d(c3 // 4, 1, 1), )
    def forward(self, x):

        f_l = self.conv_d_l1(x)

        f_l = self.conv_d_l2(f_l)

        x = self.conv(x)
        x = self.dsconv0(x)
        x = self.dsconv1(x)
        # f_l = x
        x = self.dsconv2(x)
        x = self.bottleneck(x) #32
        f_h = x
        s_d = self.conv_s_d(x)


        s_d_sig = upsample(s_d, f_l.shape[2:]).sigmoid()
        # heatmap(s_d_sig)
        s_d_sig = self.focus(s_d_sig)

        # s_d_sig_up = F.max_pool2d(s_d_sig,5,2,2)
        # s_d_sig_up = upsample(s_d_sig_up, 256)
        # heatmap(s_d_sig_up)
        # s_d_sig_up = F.conv2d(s_d_sig, self.kernel, dilation=8, padding=8, stride=1)
        # s_d_sig_up = (s_d_sig_up - s_d_sig_up.min()) / (s_d_sig_up.max() - s_d_sig_up.min() + 1e-8)
        # heatmap(s_d_sig_up)
        # s_d_sig_up = F.avg_pool2d(s_d_sig_up,8,8)
        # heatmap(s_d_sig_up)

        # s_d_sig = upsample(s_d_sig, f_l.shape[2:])
        # heatmap(f_l )
        f_l_r = s_d_sig * f_l
        e_d = self.conv_e_d(f_l_r)
        # heatmap(e_d)



        return s_d,e_d,f_l,s_d_sig,f_l_r

class focus(nn.Module):
    def __init__(self,expansion=4, dilation=4, blur=8):
        super(focus,self).__init__()
        self.kernel = torch.Tensor([[[[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]]]]).cuda()
        self.dilation = dilation
        self.expansion = expansion
        self.blur = blur
    def forward(self,x,show = False):
        size = x.shape[2]
        x = F.max_pool2d(x,self.expansion,self.expansion)
        x = upsample(x, size)
        if show:
            heatmap(x)
        x = F.conv2d(x,self.kernel,dilation=self.dilation,padding=self.dilation)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = upsample(x, size)
        if show:
            heatmap(x)
        x = F.avg_pool2d(x,self.blur,self.blur)
        x = upsample(x, size)
        if show:
            heatmap(x)
        return x

class SharedBranch(nn.Module):
    """SharedBranch"""

    def __init__(self, in_channels=96, out_channel=128, **kwargs):
        super(SharedBranch, self).__init__()
        self.bottleneck3 = _make_layer(LinearBottleneck, in_channels, out_channel, 4, 6, 1)
        self.ppm = PyramidPooling(out_channel, out_channel)

    def forward(self, r, d):
        batch = r.shape[0]
        x = torch.cat((r,d),dim=0)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        r = x[0:batch,...]
        d = x[batch:2*batch, ...]
        return r, d


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, in_channels=128, out_channel=128, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.channel_att = nn.Sequential(nn.Conv2d(2 * in_channels,2 * in_channels,1),self.relu,nn.Conv2d(2*in_channels,2*in_channels,1),self.sigmoid)
        self.channel = in_channels
        self.dsconv = _DSConv(in_channels,out_channel)
        self.score_fusion = nn.Sequential(_DSConv(2*in_channels, 2*out_channel), _DSConv(2*out_channel, out_channel ),
                      nn.Conv2d(out_channel , 1, 1), )
    def forward(self, r, d):
        # x = torch.cat((r,d),dim=1)
        # channel_att = self.channel_att(F.adaptive_avg_pool2d(x,1))
        # x = channel_att * x
        # r = x[:,0:self.channel,...]
        # d = x[:,self.channel:2*self.channel,...]

        x = torch.cat((r ,  d),dim=1)
        # x = self.dsconv(x)
        # highest_feat = x
        sal_coarse = self.score_fusion(x)
        return sal_coarse, x


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


def get_fast_scnn(dataset='citys', pretrained=False, root='./weights', map_cpu=False, **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from data_loader import datasets
    model = FastSCNN(datasets[dataset].NUM_CLASS, **kwargs)
    if pretrained:
        if(map_cpu):
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
        else:
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
    return model

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, attention, x1, x2, x3):
        # Note that there is an error in the manuscript. In the paper, the refinement strategy is depicted as ""f'=f*S1"", it should be ""f'=f+f*S1"".
        x1 = x1 + torch.mul(x1, self.upsample4(attention))
        x2 = x2 + torch.mul(x2, self.upsample2(attention))
        x3 = x3 + torch.mul(x3, attention)

        return x1, x2, x3
    
class aggregation_init(nn.Module):

    def __init__(self, channel):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = _DSConv(channel, channel)
        self.conv_upsample2 = _DSConv(channel, channel)
        self.conv_upsample3 = _DSConv(channel, channel)
        self.conv_upsample4 = _DSConv(channel, channel)
        self.conv_upsample5 = _DSConv(2 * channel, 2 * channel)

        self.conv_concat2 = _DSConv(2 * channel, 2 * channel)
        self.conv_concat3 = _DSConv(3 * channel, 3 * channel)
        self.conv4 = _DSConv(3 * channel, 3 * channel)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = x1 * x2
        x3_1 = self.conv_upsample2(self.upsample(x1)) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, x1_1), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class aggregation_final(nn.Module):

    def __init__(self, channel):
        super(aggregation_final, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = _DSConv(channel, channel)
        self.conv_upsample2 = _DSConv(channel, channel)
        self.conv_upsample3 = _DSConv(channel, channel)
        self.conv_upsample4 = _DSConv(channel, channel)
        self.conv_upsample5 = _DSConv(2 * channel, 2 * channel)

        self.conv_concat2 = _DSConv(2 * channel, 2 * channel)
        self.conv_concat3 = _DSConv(3 * channel, 3 * channel)

    def forward(self, x1, x2, x3):

        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2
class RPP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RPP, self).__init__()
        self.conv_cp = _DSConv(in_channel, out_channel)
        self.conv_1 = _DSConv(out_channel, out_channel)
        self.conv_2 = _DSConv(out_channel, out_channel)
        self.conv_3 = _DSConv(out_channel, out_channel)
        self.conv_4 = _DSConv(out_channel, out_channel)
        self.predicton = BasicConv2d(out_channel, 1, 1,activation=None)
    def forward(self, x):
        x = self.conv_cp(x)
        x1 = self.conv_1(F.avg_pool2d(x,5,1,2))
        x2 = self.conv_2(F.avg_pool2d(x1,5,1,2))
        x3 = self.conv_3(F.avg_pool2d(x2, 5, 1, 2))
        x4 = self.conv_4(F.avg_pool2d(x3, 5, 1, 2))
        x = x1 + x2 + x3 + x4 + x
        return  self.predicton(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, activation='relu'):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return  self.relu(x) if self.activation=='relu' \
        else self.sigmoid(x) if self.activation=='sigmoid' \
        else x
# if __name__ == '__main__':
#     img = torch.randn(2, 3, 512, 512).cuda()
#     depth = torch.randn(2, 3, 512, 512)
#     model = FastSal().cuda()
#     outputs = model(img,img)