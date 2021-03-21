import argparse
import os
import time
import shutil
from natsort import natsorted

def save_path():
    dir_list = os.listdir('./results/')
    dir_list = natsorted(dir_list)
    # print(dir_list)
    if len(dir_list)!=0:
        last_dir = dir_list[-1]
        # print(last_dir)
        if last_dir.split('-')[1]==time.strftime("%m") and last_dir.split('-')[2]==time.strftime("%d"): # match the month and day
            last_dir_path = './results/'+last_dir
            last_dir_file = os.listdir(last_dir_path)
            is_exist_pth = 0
            for i in last_dir_file:
                if 'pth' in i:
                    is_exist_pth = 1
                    break
            if is_exist_pth == 0:
                shutil.rmtree(last_dir_path)
                os.mkdir(last_dir_path)
                return last_dir_path
            else:
                order = last_dir_path.split('-')[-1]
                save_path = "./results/demo-%s-%s-%d" % (time.strftime("%m"), time.strftime("%d"),int(order)+1)
                os.mkdir(save_path)
                return save_path
        else: # not same day
            save_path = "./results/demo-%s-%s-0" % (time.strftime("%m"), time.strftime("%d"))
            os.mkdir(save_path)
            return save_path
    else:  # not same day
        save_path = "./results/demo-%s-%s-0" % (time.strftime("%m"), time.strftime("%d"))
        os.mkdir(save_path)
        return save_path

    # run = 0
    # save_folder = "./results/demo-%s-%s-%d" % (time.strftime("%m"), time.strftime("%d"), run)
    # while os.path.exists(save_folder) :
    #     run += 1
    #     save_folder = "./results/demo-%s-%s-%d" % (time.strftime("%m"), time.strftime("%d"), run)
    #
    # if os.path.exists(save_folder):
    #     is_exist_pth = 0
    #     for i in os.listdir(save_folder):
    #         if 'pth' in i:
    #             is_exist_pth = 1
    #     save_folder = "./results/demo-%s-%s-%d" % (time.strftime("%m"), time.strftime("%d"), run)
    #     if is_exist_pth == 0:
    #         shutil.rmtree(save_folder)
    #
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)
    # return save_folder


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--epoch', type=int, default=301, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='/home/mist/data/RGBDcollection_fast/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='/home/mist/data/RGBDcollection_fast/depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='/home/mist/data/RGBDcollection_fast/GT/', help='the training gt images root')
# parser.add_argument('--edge_root', type=str, default='D:/pytorch\data\RGBDcollection_fast/edge/', help='the training edge images root')
# parser.add_argument('--test_rgb_root', type=str, default='D:\pytorch\data/test_in_train/RGB/', help='the test rgb images root')
# parser.add_argument('--test_depth_root', type=str, default='D:\pytorch\data/test_in_train/depth/', help='the test depth images root')
# parser.add_argument('--test_gt_root', type=str, default='D:\pytorch\data/test_in_train/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default='./results/', help='the path to save models and logs')
opt = parser.parse_args()

