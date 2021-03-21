import os
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from models.net import FastSal
from data import get_loader,test_dataset
from utils import clip_gradient, adjust_lr, LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
from heatmap import heatmap
import logging
import torch.backends.cudnn as cudnn
from options import opt,save_path
import torch.nn as nn
import torch.nn.functional as F
from depth import depth

def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)

#train function
def train(train_loader, model, optimizer, epoch,save_path):

    global step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, depths,edge_gt,match_gt) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            depths=depths.cuda()
            edge_gt = edge_gt.cuda()
            match_gt = match_gt.cuda()

            cur_lr = lr_scheduler(optimizer, i, epoch)
            writer.add_scalar('learning_rate', cur_lr, global_step=(epoch-1)*total_step + i)

            out,feature_r,feature_d = model(images,depths)
            loss_f = F.binary_cross_entropy_with_logits(out[0], gts)
            p_sum = out[1].sigmoid().sum().data.cpu()/10
            n_sum = 262144 - p_sum
            beta = n_sum/p_sum
            msak_e_d = upsample(F.avg_pool2d(out[0].sigmoid().data,3,2,1), edge_gt.shape[2:])
            gt_e = edge_gt*msak_e_d
            loss_edge_d = F.mse_loss(out[1], gt_e)#, pos_weight=beta)


            loss = loss_f + 100*loss_edge_d #+ loss_edge_d + loss_edge_r #loss_coarse_r + loss_coarse_d + loss_err_r + loss_err_d +loss_edge_d
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step+=1
            epoch_step+=1
            loss_all+=loss.data
            if i % 100 == 0 or i == total_step or i==1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}, loss_final: {:.4f}, loss_c_d: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss,loss_f.data,loss_edge_d.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} '.
                    format( epoch, opt.epoch, i, total_step, loss.data))
                writer.add_scalar('Loss', loss.data, global_step=step)

                grid_image_gt = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                grid_image_depth = make_grid(depths[0].clone().cpu().data, 1, normalize=True)
                grid_image_img = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                grid_image_edge = make_grid(edge_gt[0].clone().cpu().data, 1, normalize=True)
                grid_image_edge_r = make_grid(gt_e[0].clone().cpu().data, 1, normalize=True)

                writer.add_image('input', torch.cat((grid_image_gt, grid_image_img, grid_image_depth,grid_image_edge,grid_image_edge_r), dim=2), step)

                for i in range(len(out)):
                    out[i] = upsample(out[i], (224,224))
                for i in range(len(feature_r)):
                    feature_r[i] = upsample(feature_r[i], (224,224))
                for i in range(len(feature_d)):
                    feature_d[i] = upsample(feature_d[i], (224,224))

                res = out[0][[0]].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res_final = torch.tensor(res)

                res = out[1][[0]].clone()
                res = res.data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res_c = torch.tensor(res)

                # res = out[2][[0]].clone()
                # res = res.data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # res_d = torch.tensor(res)
                #
                # res = out[3][[0]].clone()
                # res = res.data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # res_e_r = torch.tensor(res)
                #
                # res = out[4][[0]].clone()
                # res = res.data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # res_e_d =  torch.tensor(res)



                writer.add_image('out',torch.cat((res_final,res_c),dim=1), step, dataformats='HW')

                # r0 = make_grid(feature_r[0][[0]].data.cpu().permute(1,0,2,3),normalize=True,scale_each=True)
                # writer.add_image('r0',r0,step)
                # r1 = make_grid(feature_r[1][[0]].data.cpu().permute(1, 0, 2, 3), normalize=True, scale_each=True)
                # writer.add_image('r1',r1, step)
                r2 = make_grid(out[2][[0]].data.cpu().permute(1, 0, 2, 3), normalize=True, scale_each=True)
                writer.add_image('r2',r2 ,step)
                r3 = make_grid(out[3][[0]].data.cpu().permute(1, 0, 2, 3), normalize=True, scale_each=True)
                writer.add_image('r3',r3, step)
                r4 = make_grid(out[4][[0]].data.cpu().permute(1, 0, 2, 3), normalize=True, scale_each=True)
                writer.add_image('r4', r4, step)
                # d0 = make_grid(feature_d[0][[0]].data.cpu().permute(1, 0, 2, 3), normalize=True, scale_each=True)
                # writer.add_image('d0', d0, step)
                # d1 = make_grid(feature_d[1][[0]].data.cpu().permute(1, 0, 2, 3), normalize=True, scale_each=True)
                # writer.add_image('d1', d1, step)
                # d2 = make_grid(feature_d[2][[0]].data.cpu().permute(1, 0, 2, 3), normalize=True, scale_each=True)
                # writer.add_image('d2', d2, step)
                # d3 = make_grid(feature_d[3][[0]].data.cpu().permute(1, 0, 2, 3), normalize=True, scale_each=True)
                # writer.add_image('d3', d3, step)


        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 50 == 0:
            torch.save(model.state_dict(), save_path+'/epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'/epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise
        
#test function
def test(test_loader,model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt,depth, name,img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            res,_,_  = model(image,depth)
            res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum+=np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
        mae=mae_sum/test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        if epoch==1:
            best_mae=mae
        else:
            if mae<best_mae:
                best_mae=mae
                best_epoch=epoch
                torch.save(model.state_dict(), save_path+'/epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))
 
if __name__ == '__main__':

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        print('USE GPU 0,1')
    cudnn.benchmark = True

    # build the model
    model = depth()

    # if (opt.load is not None):
    #     model.load_state_dict(torch.load(opt.load),strict=False)
    #     print('load model from ', opt.load)
    # model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('./results/demo-01-15-0/epoch_best.pth'),strict=False)

    model.cuda()
    params = model.parameters()
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(num_params)
    # optimizer = torch.optim.Adadelta(filter(lambda p:p.requires_grad,model.parameters()), 1,weight_decay=0.0005)#opt.lr)
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),opt.lr)

    # set the path
    image_root = opt.rgb_root
    gt_root = opt.gt_root
    depth_root = opt.depth_root
    edge_root = opt.edge_root
    test_image_root = opt.test_rgb_root
    test_gt_root = opt.test_gt_root
    test_depth_root = opt.test_depth_root
    save_path = save_path()
    # load data
    print('load data...')
    train_loader = get_loader(image_root, gt_root, depth_root,edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
    total_step = len(train_loader)
    lr_scheduler = LR_Scheduler('poly', opt.lr, opt.epoch, total_step)

    logging.basicConfig(filename=save_path + '/log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Train")
    logging.info("Config")
    logging.info(
        'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
            opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
            opt.decay_epoch))

    # set loss function
    #CE = torch.nn.BCEWithLogitsLoss()

    step = 0
    writer = SummaryWriter(save_path + '/summary')
    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        #print(model)
        # cur_lr=adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch,save_path)
        if  epoch%50==0:
            test(test_loader,model,epoch,save_path)
