import numpy as np
import cv2
import torch
import math
import torch.nn.functional as F

def heatmap(x_show,img=None,name=None,multi_channel=True):
    if multi_channel:
        nmaps = x_show.size(1)
        if nmaps>60 :
            x_show = F.interpolate(x_show, 100, mode='bilinear', align_corners=True)
        else:
            x_show = F.interpolate(x_show, 200, mode='bilinear', align_corners=True)
        x_show = x_show.permute(1,0,2,3)
        padding=2
        pad_value = 1

        xmaps = min(8, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
        height, width = int(x_show.size(2) + padding), int(x_show.size(3) + padding)
        num_channels = x_show.size(1)
        grid = x_show.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                # x_show.copy_() is a valid method but seems to be missing from the stubs
                # https://pytorch.org/docs/stable/x_shows.html#torch.x_show.copy_
                grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                    2, x * width + padding, width - padding
                ).copy_(x_show[k])
                k = k + 1
        x_show = grid.data.cpu().numpy().squeeze()

    else:
        x_show = torch.mean(x_show, dim=1, keepdim=True).data.cpu().numpy().squeeze()
        x_show = cv2.resize(x_show, (320, 320))
    x_show = (x_show - x_show.min()) / (x_show.max() - x_show.min() + 1e-8)
    if img!=None:
        img = img.data.cpu().numpy().squeeze()
        img = img.transpose((1,2,0))
        img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        img = img[:,:,::-1]
        #img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.uint8(255 * img)
        img = cv2.resize(img, (320, 320))
    # x_show = np.uint8(255 * x_show)
    # x_show = cv2.applyColorMap(x_show, cv2.COLORMAP_JET)


    # print(x_show.shape,img.shape)
    if img!=None:
        x_show = cv2.addWeighted(img,0.5,x_show,0.5,0)
    if name!=None:
        cv2.imwrite('D:\pytorch\data\example\heatmap/' +name+'.jpg',x_show)
    else:
        cv2.imshow('img', x_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
