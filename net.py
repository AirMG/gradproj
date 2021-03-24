import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mobilenet import MobileNetV2Encoder
import time
# from heatmap import  heatmap
import torchvision.models as models
# from .unet import Unet

# delete bts and use simple decoder
INPUT_SIZE = 512

class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, bias=False,stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=bias),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

def merge(x,y):
    return x+y,x+y

def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)

def initialize_weights(model):
    print('test-func')
    m = models.mobilenet_v2(pretrained=False)
    # print('test-func')
    pthfile = r'/home/mist/gradproj/mobilenet_v2-b0353104.pth'
    # m = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    m.load_state_dict(torch.load(pthfile))
    pretrained_dict = m.state_dict()
    all_params = {}
    for k, v in model.state_dict().items():
        if k in pretrained_dict.keys() and v.shape==pretrained_dict[k].shape:
            v = pretrained_dict[k]
            all_params[k] = v
    model.load_state_dict(all_params,strict=False)

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

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

def softmax_2d(x):
    return torch.exp(x) / torch.sum(torch.sum(torch.exp(x), dim=-1, keepdim=True), dim=-2, keepdim=True)


class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FastSal(nn.Module):
    def __init__(self):
        super(FastSal, self).__init__()
        self.depth_pre = MobileNetV2Encoder(1).layer1
        self.rgb = MobileNetV2Encoder(3)
        initialize_weights(self.rgb)

        self.ppmx = PyramidPooling(320,64)
        self.ppmy = PyramidPooling(320, 64)

        lowOutputChannels = 64

        # Prediction head
        self.last_conv = nn.Sequential(
                nn.Conv2d(2*lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(lowOutputChannels),
                nn.ReLU(),

                nn.Conv2d(lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(lowOutputChannels),
                nn.ReLU(),

                nn.Conv2d(lowOutputChannels, 1, kernel_size=1, stride=1)
            )

        self.last_conv_rgb = nn.Sequential(
            nn.Conv2d(2*lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(lowOutputChannels),
            nn.ReLU(),

            nn.Conv2d(lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(lowOutputChannels),
            nn.ReLU(),

            nn.Conv2d(lowOutputChannels, 1, kernel_size=1, stride=1)
        )

        self.last_conv_depth = nn.Sequential(
            nn.Conv2d(2*lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(lowOutputChannels),
            nn.ReLU(),

            nn.Conv2d(lowOutputChannels, lowOutputChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(lowOutputChannels),
            nn.ReLU(),

            nn.Conv2d(lowOutputChannels, 1, kernel_size=1, stride=1)
        )

        # low_level_feature to 48 channels
        self.rgb_conv1_cp = BasicConv2d(16, lowOutputChannels,1)
        self.depth_conv1_cp = BasicConv2d(16, lowOutputChannels,1)
        self.rgb_layer1_cp = BasicConv2d(24, lowOutputChannels, 1)
        self.depth_layer1_cp = BasicConv2d(24, lowOutputChannels, 1)
        self.rgb_layer2_cp = BasicConv2d(32, lowOutputChannels, 1)
        self.depth_layer2_cp = BasicConv2d(32, lowOutputChannels, 1)
        self.rgb_layer3_cp = BasicConv2d(96, lowOutputChannels, 1)
        self.depth_layer3_cp = BasicConv2d(96, lowOutputChannels, 1)
        self.rgb_layer4_cp = BasicConv2d(320, lowOutputChannels, 1)
        self.depth_layer4_cp = BasicConv2d(320, lowOutputChannels, 1)

        self.fusion_high = BasicConv2d(2*lowOutputChannels,lowOutputChannels,3,1,1)
        self.fusion_low = BasicConv2d(2 * lowOutputChannels, lowOutputChannels, 3, 1, 1)
        # self.fusion = BasicConv2d(2 * lowOutputChannels, lowOutputChannels, 3, 1, 1)

        # self.


    def forward(self, img, depth):
        x = self.rgb.layer1(img)
        conv1_feat = x
        y = self.depth_pre(depth)
        depth_conv1_feat = y
        # a = torch.max(x, dim=1, keepdim=True)[0]
        # b = torch.max(y, dim=1, keepdim=True)[0]
        # heatmap(a)
        # heatmap(b)
        # heatmap(x*y)
        # heatmap(torch.mean(x*y/(x+y), dim=1, keepdim=True))
        x, y = merge(x, y)
        after_depth_conv1_feat = y
        after_conv1_feat = x

        f = torch.cat((x, y), dim=0)
        f = self.rgb.layer2(f)
        x = f[0:f.shape[0] // 2, ...]
        y = f[f.shape[0]//2:f.shape[0],...]
        layer1_feat = x
        depth_layer1_feat = y
        x, y = merge(x, depth_layer1_feat)
        after_layer1_feat = x
        after_depth_layer1_feat = y
        low_level_feature = x
        low_level_depth_feature = y

        f = torch.cat((x, y), dim=0)
        f = self.rgb.layer3(f)
        x = f[0:f.shape[0] // 2, ...]
        y = f[f.shape[0] // 2:f.shape[0], ...]
        layer2_feat = x
        depth_layer2_feat = y
        x, y = merge(x, depth_layer2_feat)
        after_layer2_feat = x
        after_depth_layer2_feat =y

        f = torch.cat((x, y), dim=0)
        f = self.rgb.layer4(f)
        x = f[0:f.shape[0] // 2, ...]
        y = f[f.shape[0] // 2:f.shape[0], ...]
        layer3_feat = x
        depth_layer3_feat = y
        x, y = merge(x,y)
        after_layer3_feat = x
        after_depth_layer3_feat = y

        f = torch.cat((x, y), dim=0)
        f = self.rgb.layer5(f)
        x = f[0:f.shape[0] // 2, ...]
        y = f[f.shape[0] // 2:f.shape[0], ...]
        layer4_feat = x
        depth_layer4_feat = y
        x, y = merge(x, depth_layer4_feat)
        after_layer4_feat = x
        after_depth_layer4_feat = y


        img_feat_lst = [conv1_feat, layer1_feat, layer2_feat, layer3_feat, layer4_feat]
        img_feat_attentioned_lst = [after_conv1_feat, after_layer1_feat, after_layer2_feat, after_layer3_feat, after_layer4_feat]
        depth_feat_lst = [depth_conv1_feat, depth_layer1_feat, depth_layer2_feat, depth_layer3_feat, depth_layer4_feat]
        depth_feat_attentioned_lst = [after_depth_conv1_feat, after_depth_layer1_feat, after_depth_layer2_feat, after_depth_layer3_feat,
                                    after_depth_layer4_feat]

            #
            # x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)
            # y = F.upsample(y, scale_factor=4, mode='bilinear', align_corners=True)

        x = self.ppmx(x)
        y = self.ppmy(y)

        x_aspp = x
        img_feat_lst.append(x)
        y_aspp = y
        depth_feat_lst.append(y)

        after_layer4_feat = self.rgb_layer4_cp(after_layer4_feat)
        after_depth_layer4_feat = self.depth_layer4_cp(after_depth_layer4_feat)
        
        after_layer3_feat = self.rgb_layer3_cp(after_layer3_feat)
        after_depth_layer3_feat = self.depth_layer3_cp(after_depth_layer3_feat)
        after_layer2_feat = F.upsample(self.rgb_layer2_cp(after_layer2_feat), scale_factor=4, mode='bilinear',
                   align_corners=True)
        after_depth_layer2_feat = F.upsample(self.depth_layer2_cp(after_depth_layer2_feat), scale_factor=4, mode='bilinear',
                   align_corners=True)
        after_layer1_feat = F.upsample(self.rgb_layer1_cp(after_layer1_feat), scale_factor=2, mode='bilinear',
                   align_corners=True)
        after_depth_layer1_feat = F.upsample(self.depth_layer1_cp(after_depth_layer1_feat), scale_factor=2, mode='bilinear',
                   align_corners=True)
        after_conv1_feat = self.rgb_conv1_cp(after_conv1_feat)
        after_depth_conv1_feat = self.depth_conv1_cp(after_depth_conv1_feat)

        # produce rgb/d-low/high feature
        rgb_high_feat = x_aspp + after_layer4_feat +after_layer3_feat
        depth_high_feat = y_aspp + after_depth_layer4_feat + after_depth_layer3_feat
        rgb_low_feat = after_layer2_feat+after_layer1_feat+after_conv1_feat
        depth_low_feat = after_depth_layer2_feat+after_depth_layer1_feat+after_depth_conv1_feat

        #rgb_fusion
        # RGB branch-concatenate
        rgb_feat = torch.cat((F.upsample(rgb_high_feat, scale_factor=8, mode='bilinear',align_corners=True), rgb_low_feat), dim=1)
        # Prediction head
        sal_rgb = self.last_conv_rgb(rgb_feat)
        sal_rgb = F.upsample(sal_rgb, img.size()[2:], mode='bilinear', align_corners=True)

        #depth_fusion
        # Depth branch-concatenate
        depth_feat = torch.cat((F.upsample(depth_high_feat, scale_factor=8, mode='bilinear',
                   align_corners=True), depth_low_feat), dim=1)
        sal_depth = self.last_conv_depth(depth_feat)
        sal_depth = F.upsample(sal_depth, img.size()[2:], mode='bilinear', align_corners=True)


        #final_fusion
        # 64 Dimension
        higt_feat = self.fusion_high(
            torch.cat((rgb_high_feat,depth_high_feat), dim=1))
        low_feat = self.fusion_low(
            torch.cat((rgb_low_feat , depth_low_feat), dim=1))
        
        high_feat = F.upsample(higt_feat, scale_factor=8, mode='bilinear',
                   align_corners=True)
        sal = self.last_conv(torch.cat((high_feat,low_feat),dim=1))
        sal = F.upsample(sal, img.size()[2:], mode='bilinear', align_corners=True)

        return sal, sal_rgb, sal_depth

if __name__ == '__main__':
    img = torch.randn(1, 3, 256, 256).cuda()
    depth = torch.randn(1, 1, 256, 256).cuda()
    model = FastSal().cuda()
    model.eval()
    torch.cuda.synchronize()
    time1= time.time()
    for i in range(1000):
        outputs = model(img,depth)
    time2 = time.time()
    torch.cuda.synchronize()
    print(1000/(time2-time1))