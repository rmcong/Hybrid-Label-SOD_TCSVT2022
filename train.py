#!/usr/bin/python3
#coding=utf-8

import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

import datetime
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data import dataset
from net import WSLNet_UP, WSLNet_DOWN
import logging as logger
from lib.data_prefetcher import DataPrefetcher
import numpy as np
from net import MINet_ResNet50
import lib.LR_Scheduler as sche
import lib.CEL as CEL

TAG = "ours"
SAVE_PATH = "ours"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")



BASE_LR = 1e-3
MAX_LR = 0.1
FIND_LR = False #True
IMAGE_GROUP = 0


def train(Dataset, Network1, Network2, Network3):
    ## dataset R-Net
    cfg    = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode='train_' + str(IMAGE_GROUP), batch=8, lr=1e-4, momen=0.9, decay=5e-4, epoch=30)
    ## dataset S-Net
    cfg2   = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode='train_' + str(IMAGE_GROUP), batch=8, lr=1e-4, momen=0.9, decay=5e-4, epoch=30)

    ## network
    net1 = Network1(cfg)
    net2 = Network2(cfg)
    net  = Network3(cfg2)
    ##print('net=', net)
    net1.cuda()
    net2.cuda()
    net.cuda()
    ## parameter
    base, head = [], []

    params = [
        {
            "params": [
                p for name, p in net.named_parameters() if ("bias" in name or "bn" in name)
            ],
            "weight_decay": 0,
        },
        {
            "params": [
                p
                for name, p in net.named_parameters()
                if ("bias" not in name and "bn" not in name)
            ]
        },
    ]
    optimizer2 = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.decay)

    for name, param in net1.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer3  = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg2.lr, momentum=cfg2.momen, weight_decay=cfg2.decay, nesterov=True)



    for name, param in net2.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer1 = torch.optim.Adam([{'params': base}, {'params': head}], lr=cfg2.lr, weight_decay=cfg2.decay)

    sw          = SummaryWriter(cfg.savepath)
    global_step = 0

    mae_global = 1
    mae_snet = 1
    for epoch_g in range(5):
        if epoch_g == 0:
            mode_refine1 = 'train_0'
            mode_refine2 = 'train_0'
        elif epoch_g == 1:
            mode_refine1 = 'train_2'
            mode_refine2 = 'train_0'
        elif epoch_g == 2:
            mode_refine1 = 'train_2_4'
            mode_refine2 = 'train_0'
        elif epoch_g == 3:
            mode_refine1 = 'train_2_4_6'
            mode_refine2 = 'train_0'
        elif epoch_g == 4:
            mode_refine1 = 'train_2_4_6_8'
            mode_refine2 = 'train_0'
        elif epoch_g == 5:
            mode_refine1 = 'train_2_4_6_8_10'
            mode_refine2 = 'train_0'
        else:
            mode_refine1 = 'train_2_4_6_8_9'
            mode_refine2 = 'train_0'
        cfg1 = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode=mode_refine1, batch=8, lr=1e-4,
                                  momen=0.9, decay=5e-4, epoch=30)
        cfg2 = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode=mode_refine2, batch=8, lr=1e-4,
                              momen=0.9, decay=5e-4, epoch=30)
        data1 = Dataset.Data(cfg1)
        data2 = Dataset.Data(cfg2)
        loader1 = DataLoader(data1, batch_size=cfg1.batch, shuffle=True, num_workers=8, drop_last=True)
        loader2 = DataLoader(data2, batch_size=cfg2.batch, shuffle=True, num_workers=8, drop_last=True)
        db_size1 = len(loader1)
        db_size2 = len(loader2)
        if epoch_g == 0:
            scheduler1 = sche.make_scheduler(optimizer=optimizer1, total_num=cfg1.epoch*db_size1+cfg2.epoch*db_size2, scheduler_type="poly_warmup",
                                            scheduler_info=dict(lr_decay=0.9, warmup_epoch=2500),
                                            )
            scheduler2 = sche.make_scheduler(optimizer=optimizer3, total_num=cfg1.epoch*db_size1+cfg2.epoch*db_size2,
                                             scheduler_type="poly_warmup",
                                             scheduler_info=dict(lr_decay=0.9, warmup_epoch=2500),
                                             )
        else:
            scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[10, 20], gamma=0.1)
            scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer3, milestones=[10, 20], gamma=0.1)

        net2.train()

        for epoch in range(cfg1.epoch):
            prefetcher = DataPrefetcher(loader1)
            logger.info(cfg1.mode)
            batch_idx = -1
            image, mask, coarse = prefetcher.next()
            f = 0
            while image is not None:
                optimizer1.zero_grad()
                optimizer3.zero_grad()
                lr = scheduler1.get_lr()[0]
                optimizer1.param_groups[0]['lr'] = 0.1 * lr
                optimizer1.param_groups[1]['lr'] = lr
                batch_idx += 1
                global_step += 1


                out2_c, out3_c, out4_c, out5_c, down_out1, down_out2, down_out3, down_out4, down_out5 = net2(image)
                out2, out3, out4, out5 = net1(image, coarse, down_out1, down_out2, down_out3, down_out4, down_out5)

                loss2 = F.binary_cross_entropy_with_logits(out2, mask)
                loss3 = F.binary_cross_entropy_with_logits(out3, mask)
                loss4 = F.binary_cross_entropy_with_logits(out4, mask)
                loss5 = F.binary_cross_entropy_with_logits(out5, mask)
                loss2c = F.binary_cross_entropy_with_logits(out2_c, mask)
                loss3c = F.binary_cross_entropy_with_logits(out3_c, mask)
                loss4c = F.binary_cross_entropy_with_logits(out4_c, mask)
                loss5c = F.binary_cross_entropy_with_logits(out5_c, mask)
                loss6  = loss2c*1 + loss3c*0.8 + loss4c*0.6 + loss5c*0.4

                loss = loss2*1 + loss3*0.8 + loss4*0.6 + loss5*0.4 + loss6*0.4

                if f==0:
                    loss = 0.5*loss

                loss.backward()
                optimizer1.step()
                optimizer3.step()
                if epoch_g == 0:
                    scheduler1.step()
                    scheduler2.step()

                sw.add_scalar('lr', optimizer1.param_groups[0]['lr'], global_step=global_step)
                sw.add_scalars('loss',
                               {'loss2': loss2.item(), 'loss3': loss3.item(), 'loss4': loss4.item(), 'loss5': loss5.item(),
                                'loss6': loss6.item(), 'loss': loss.item()}, global_step=global_step)
                if batch_idx % 10 == 0:
                    msg = '%s | step:%d/%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f | loss6=%.6f ' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, epoch_g+1 ,optimizer1.param_groups[0]['lr'],
                    loss.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item())
                    print(msg)
                    logger.info(msg)
                image, mask, coarse = prefetcher.next()
                if f == 0 and image is None:
                    f = 1
                    prefetcher = DataPrefetcher(loader2)
                    logger.info(cfg2.mode)
                    image, mask, coarse = prefetcher.next()

            if epoch_g != 0:
                scheduler1.step()
                scheduler2.step()
            if (epoch + 1) % 10 == 0:
                torch.save(net1.state_dict(),
                           cfg.savepath + '/model-refine-up-' + str(epoch_g) + '-' + str(epoch + 1) + '.pt', _use_new_zipfile_serialization=False)
                torch.save(net2.state_dict(),
                           cfg.savepath + '/model-refine-down-' + str(epoch_g) + '-' + str(epoch + 1) + '.pt', _use_new_zipfile_serialization=False)


        #test dataset
        cfg_mae = Dataset.Config(datapath='./data/SOD', mode='test0')
        data_mae = Dataset.Data(cfg_mae)
        loader_mae = DataLoader(data_mae, batch_size=1, shuffle=True, num_workers=8)
        net_mae1 = net1
        net_mae1.train(False)
        net_mae1.eval()
        net_mae2 = net2
        net_mae2.train(False)
        net_mae2.eval()

        with torch.no_grad():
            mae, cnt = 0, 0
            for image, mask, coarse, (H, W), name in loader_mae:
                image, coarse, mask = image.cuda().float(), coarse.cuda().float(), mask.cuda().float()
                out2_c, out3_c, out4_c, out5_c, down_out1, down_out2, down_out3, down_out4, down_out5 = net2(image)
                out2, out3, out4, out5 = net1(image, coarse, down_out1, down_out2, down_out3, down_out4, down_out5)
                pred = torch.sigmoid(out2)
                #mae
                cnt += 1
                mae += (pred - mask).abs().mean()

            mae = mae/cnt
            mae = mae.cpu().item()
            print("mae=", mae)
            logger.info("mae=" + str(mae))

            if mae > mae_global:
                if epoch_g == 0:
                    refine_test = 'test_0'
                elif epoch_g == 1:
                    refine_test = 'test_3'
                elif epoch_g == 2:
                    refine_test = 'test_5'
                elif epoch_g == 3:
                    refine_test = 'test_7'
                elif epoch_g == 4:
                    refine_test = 'test_9'
                else:
                    refine_test = 'test_10'

                net1.load_state_dict(torch.load("ours/model-refine-up-" + str(epoch_g - 1) + "-30.pt"))
                net2.load_state_dict(torch.load("ours/model-refine-down-" + str(epoch_g-1) + "-30.pt"))
                print("model not change")
                logger.info("model not change")
            else:
                if epoch_g == 0: refine_test = 'test_0'
                elif epoch_g == 1: refine_test = 'test_1_3'
                elif epoch_g == 2: refine_test = 'test_1_3_5'
                elif epoch_g == 3:
                    refine_test = 'test_1_3_5_7'
                elif epoch_g == 4:
                    refine_test = 'test_1_3_5_7_9'
                else:
                    refine_test = 'test_1_3_5_7_9_10'
                mae_global = mae
                print("model change")
                logger.info("model change")
        cfg_test  = Dataset.Config(datapath='./data/DUTS', mode=refine_test)
        data_test = Dataset.Data(cfg_test)
        loader_test = DataLoader(data_test, batch_size=1, shuffle=True, num_workers=8)

        #test network
        net_test1 = net1
        net_test1.train(False)
        net_test.cuda()
        net_test1.eval()

        net_test2 = net2
        net_test2.train(False)
        net_test.cuda()
        net_test2.eval()

        with torch.no_grad():
            i = 0
            for image, mask, coarse, (H, W), name in loader_test:
                image, coarse          = image.cuda().float(), coarse.cuda().float()
                out2_c, out3_c, out4_c, out5_c, down_out1, down_out2, down_out3, down_out4, down_out5 = net2(image)
                out2, out3, out4, out5 = net1(image, coarse, down_out1, down_out2, down_out3, down_out4, down_out5)
                out2 = F.interpolate(out2, size=(H,W), mode='bilinear')
                pred = (torch.sigmoid(out2[0, 0]) * 255).cpu().numpy()
                head = './data/DUTS/noise' + str(epoch_g)
                if not os.path.exists(head):
                    os.makedirs(head)

                cv2.imwrite(head + '/' + name[0], np.uint8(pred))
                name = name[0].split('.')
                if i == 0:
                    cv2.imwrite('./data/DUTS/log_coarse/' + name[0] + '_' + str(epoch_g) + '_refine.png', np.uint8(pred))
                cv2.imwrite('./data/DUTS/coarse/' + name[0] + '.png', np.uint8(pred))
                i += 1

        ## dataset MINet
        if epoch_g == 0:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1'
        elif epoch_g == 1:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3'
        elif epoch_g == 2:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3_5'
        elif epoch_g == 3:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3_5_7'
        elif epoch_g == 4:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3_3_5_7_9'
        elif epoch_g == 5:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3_3_5_7_9_10'
        else:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3_3_5_7_9_10'
        cfg0 = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode=mode_sod0, batch=12, lr=1e-4, momen=0.9,
                                 decay=5e-4, epoch=30)
        cfg1 = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode=mode_sod1, batch=12, lr=1e-4, momen=0.9,
                              decay=5e-4, epoch=30)
        data0 = Dataset.Data(cfg0)
        data1 = Dataset.Data(cfg1)
        loader0 = DataLoader(data0, batch_size=cfg0.batch, shuffle=True, num_workers=8)
        loader1 = DataLoader(data1, batch_size=cfg1.batch, shuffle=True, num_workers=8)
        db_size0 = len(loader0)
        db_size1 = len(loader1)
        if epoch_g ==0:
            scheduler = sche.make_scheduler(optimizer=optimizer2, total_num=cfg0.epoch*db_size0 + cfg1.epoch*db_size1, scheduler_type="poly_warmup",
                                            scheduler_info=dict(lr_decay=0.9, warmup_epoch=2500),
                                            )
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[10, 20], gamma=0.1)
        #training1
        net.train()
        global_step = 0
        for epoch in range(cfg0.epoch):
            batch_idx = -1
            prefetcher = DataPrefetcher(loader1)
            logger.info(cfg1.mode)
            image, mask, coarse = prefetcher.next()
            f = 0
            while image is not None:
                global_step += 1
                batch_idx += 1

                optimizer2.zero_grad()

                #train net
                train_pred = net(image)
                #loss function
                loss_func1 = torch.nn.BCEWithLogitsLoss(reduction="mean").cuda()
                loss_func2 = CEL.CEL().cuda()
                loss1 = loss_func1(train_pred, mask)
                loss2 = loss_func2(train_pred, mask)
                loss  = loss1 + loss2
                if f == 0:
                    loss = 0.5*loss

                loss.backward()
                optimizer2.step()
                if epoch_g == 0:
                    scheduler.step()

                if batch_idx % 10 == 0:
                    msg = '%s | step:%d/%d/%d/%d | lr=%.6f | loss=%.6f | loss1=%.6f | loss2=%.6f ' % (datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, epoch_g+1, optimizer2.param_groups[0]['lr'], loss.item(), loss1.item(), loss2.item())
                    print(msg)
                    logger.info(msg)
                image, mask, coarse = prefetcher.next()
                if f == 0 and image is None:
                    f = 1
                    prefetcher = DataPrefetcher(loader0)
                    logger.info(cfg0.mode)
                    image, mask, coarse = prefetcher.next()
            if epoch_g != 0:
                scheduler.step()
            if (epoch+1) % 10 == 0 or (epoch+1) == cfg.epoch:
                torch.save(net.state_dict(), cfg.savepath+'/model-sod-' + str(epoch_g) + '-' + str(epoch+1) + '.pt', _use_new_zipfile_serialization=False)

        # test dataset
        cfg_mae = Dataset.Config(datapath='./data/DUTS', mode='test0')
        data_mae = Dataset.Data(cfg_mae)
        loader_mae = DataLoader(data_mae, batch_size=1, shuffle=True, num_workers=8)
        net_mae = net
        net_mae.train(False)
        net_mae.eval()
      

        with torch.no_grad():
            mae, cnt = 0, 0
            for image, mask, coarse, (H, W), name in loader_mae:
                image, coarse, mask = image.cuda().float(), coarse.cuda().float(), mask.cuda().float()
                pred_mask = net_mae(image)
                pred = torch.sigmoid(pred_mask)
                #mae
                cnt += 1
                mae += (pred - mask).abs().mean()
            mae = mae/cnt
            mae = mae.cpu().item()
            print("mae_snet=", mae)
            logger.info("mae_snet=" + str(mae))

        if mae > mae_snet:
                if epoch_g == 0:
                    sod_test = 'test_2'
                elif epoch_g == 1:
                    sod_test = 'test_4'
                elif epoch_g == 2:
                    sod_test = 'test_6'
                elif epoch_g == 3:
                    sod_test = 'test_8'
                elif epoch_g == 4:
                    sod_test = 'test_10'
                else:
                    sod_test = 'test_10'

                net.load_state_dict(torch.load("ours/model-sod-" + str(epoch_g - 1) + "-30.pt"))
                print("model not change")
                logger.info("model not change")
        else:
            if epoch_g == 0: sod_test = 'test_2'
            elif epoch_g == 1: sod_test = 'test_2_4'
            elif epoch_g == 2: sod_test = 'test_2_4_6'
            elif epoch_g == 3:
                sod_test = 'test_2_4_6_8'
            elif epoch_g == 4:
                sod_test = 'test_2_4_6_8_10'
            mae_snet = mae
            print("model change")
            logger.info("model change")

        cfg_test = Dataset.Config(datapath='./data/DUTS', mode=sod_test)
        data_test = Dataset.Data(cfg_test)
        loader_test = DataLoader(data_test, batch_size=1, shuffle=True, num_workers=8)
        # test network
        net_test = net
        net_test.train(False)
        net_test.cuda()
        net_test.eval()
      
        with torch.no_grad():
            i = 0
            for image, mask, coarse, (H, W), name in loader_test:
                image                  = image.cuda().float()
                pred_mask = net_test(image)
                pred_mask = F.interpolate(pred_mask, size=(H,W), mode='bilinear')
                pred = (torch.sigmoid(pred_mask[0, 0]) * 255).cpu().numpy()
                name = name[0].split('.')
                head = './data/DUTS/coarse_sod_' + str(epoch_g)
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.uint8(pred))
                cv2.imwrite('./data/DUTS/coarse/' + name[0] + '.png', np.uint8(pred))
                if i == 0:
                       cv2.imwrite('./data/DUTS/log_coarse/' + name[0] + '_' + str(epoch_g) + '_sod.png', np.uint8(pred))
                i += 1

if __name__=='__main__':
    train(dataset, WSLNet_UP, WSLNet_DOWN, MINet_ResNet50)
