import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from heatmap import heatmap
from net import FastSal
from data import test_dataset
import time
from evaluation import fast_evaluation


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/home/mist/data/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
elif opt.gpu_id == '3':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print('USE GPU 3')
elif opt.gpu_id=='all':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print('USE GPU 0,1,2,3')

#load the model
model = FastSal()
# model = torch.nn.DataParallel(model)
# Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
model.load_state_dict(torch.load('./results/demo-03-24-0/epoch_300.pth')) #'./model_pths/BBSNet_epoch_best.pth'))
model.cuda()
model.eval()

#test


def save(res,gt,notation=None,sigmoid=True):
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze() if sigmoid ==True else res.data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    # print('save img to: ', os.path.join(save_pasth, name.replace('.png','_'+notation+'.png') if notation != None else name))
    cv2.imwrite(os.path.join(save_path, name.replace('.png','_'+notation+'.png') if notation != None else name), res * 255)

test_datasets = [ 'NJU2K','NLPR','STERE', 'RGBD135', 'LFSD','SIP']# ['NJU2K','NLPR','STERE', 'RGBD135', 'LFSD','SIP']
for dataset in test_datasets:
    with torch.no_grad():
        save_path = './results/test-03-24-without_bts-300pth/' + dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/RGB/'
        gt_root = dataset_path + dataset + '/GT/'
        depth_root=dataset_path +dataset +'/depth/'
        test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
        sum = 0
        avg_fps = 0
        sum_fps = 0
        for i in range(test_loader.size):
            image, gt,depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            torch.cuda.synchronize()
            time_s = time.time()
            out,r,d= model(image,depth)
            torch.cuda.synchronize()
            time_e = time.time()
            if(i%200==0):
                print('Speed: %f FPS' % (1 / (time_e - time_s)))
            sum_fps += 1 / (time_e - time_s)
            save(out, gt)
            save(r, gt, 'r')
            save(d, gt, 'd')
            # save(out[3], gt, 'er')
            # save(out[4], gt, 'ed')
            # save(out[3]*out[4], gt, 'ec')
            # save(out[0],gt,str(out[4].data.cpu().numpy()))
            # save(out[1], gt, 'sf')
            # save(out[2], gt, 'ser')
            # save(out[3], gt, 'scd')
            # # a= edge_r.sigmoid()
            # b=edge_d[0].sigmoid()
            # score = ((a*b).sum()/(a.sum()+b.sum())).data.cpu().numpy()
            # sum = sum+score
            # score = str(format(score*1000,'.0f'))
            # save(a*b,gt,'score_'+score)
        # print(sum/i)
        print('avg Speed: %f FPS' % (sum_fps/test_loader.size))
    fast_evaluation.main(save_path, dataset, os.path.split(save_path)[0])
    print('Test Done!')
