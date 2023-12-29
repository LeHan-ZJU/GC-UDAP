import argparse
import logging
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.backends import cudnn
import torch.distributed as dist

from GCN.gcn import Net

from Models.eval_pose import eval_net_s2_update
from Models.ContrastModel import ContrastNet1_MultiA, ResSelf_pre
from Loss.GraphLoss import graph_contrative_loss
from Loss.ContrastiveLoss import self_contrative_loss, self_contrative_meanloss
from Tools.process import Combine_csv
from Tools.inference_block import inference_PoseNet

from utils.dataset_csv import DatasetStage2_iteration
from torch.utils.data import DataLoader, random_split

torch.distributed.init_process_group(backend="nccl")
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

num_points = 6

resize_w = 640
resize_h = 480  # 128 64 32 16 8
extract_list = ["layer4"]


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target heatmaps',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cls_number', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--number', type=int, default=1)
    parser.add_argument('--pretrained', default=True, action='store_false')
    parser.add_argument('--anchor_number', '-a', type=int, default=49)

    parser.add_argument('-au', '--augment', dest='augment', type=float, default=0, help='Augmenting or not')
    parser.add_argument('-an', '--angle', dest='angle', type=float, default=180, help='Angle of self contrastive')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch', metavar='B', dest='batch', type=int, nargs='?', default=6, help='Batch size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-5, dest='lr',
                        help='Learning rate')
    parser.add_argument('-f', '--load', dest='load', type=str,  # default=False,
                        default='./TrainedModels/ContrastiveModels/train_ResSelf_s2_GraphConstrint/training_paralle_GCNv2/CP_epoch30.pth',
                        help='Load model from a .pth file')
    parser.add_argument('-w', '--weight', dest='weight', type=float, default=0, help='the weight of similar loss')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=5.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-s', '--source_label', dest='source_label', type=str,
                        default='/8T/hanle/Datasets/RatPose/label/RatPoseLabels/RatPoseAll_indoor_trainval.csv',
                        # default='/8T/hanle/Datasets/RatPose/label/RatPoseLabels/RatPoseAll_indoor_crop_v2_trainval.csv',
                        # default='/8T/hanle/Datasets/RatPose/label/RatPoseLabels/VerticalTilt_crop_v2_trainval.csv',
                        # default='/8T/hanle/Datasets/RatPose/label/RatPoseLabels/Vertical_Indoor_label_new_crop_v2_trainval.csv',
                        # default='/8T/hanle/Datasets/TigDog/behaviorDiscovery2.0/landmarks/tiger/tiger_trainval.csv',
                        # default='/8T/hanle/Datasets/TigDog/behaviorDiscovery2.0/landmarks/tiger/tiger_trainval.csv',
                        help='Label path of source image label')
    parser.add_argument('-t', '--target_label', dest='target_label', type=str,
                        default='./results/pretrain_ResSelf/debug1/VerticalTilt_all_GCNv2/debuggraph_refine/Right/Pesudo-label3.csv',
                        help='Label path of source image label')
    parser.add_argument('-tu', '--target_unlabel', dest='target_unlabel', type=str,
                        default='./results/pretrain_ResSelf/debug1/VerticalTilt_all_GCNv2/debuggraph_refine/Right/noPesudo-label.csv',
                        help='Label path of source image label')
    parser.add_argument('-i', '--imgs', dest='Imgs', type=str,
                        # default='/8T/hanle/Datasets/TigDog/behaviorDiscovery2.0/',
                        default='/8T/hanle/Datasets/RatPose/all/',
                        help='File path of images')
    parser.add_argument('-pb', '--path_backbone', dest='path_backbone', type=str,
                        default='./TrainedModels/ContrastiveModels/pretrain_ResSelf/debug1/Resnet50_epoch300.pth',
                        help='the path of backbone')
    parser.add_argument('-pt', '--path_transformer', dest='path_transformer', type=str,
                        default='./TrainedModels/ContrastiveModels/pretrain_ResSelf/debug1/TransformerPart_epoch300.pth',
                        help='the path of transformer part')
    parser.add_argument('-ph', '--path_head', dest='path_head', type=str,
                        default='./TrainedModels/ContrastiveModels/pretrain_ResSelf/debug1/PoseHead_epoch300.pth',
                        help='the path of head part')
    parser.add_argument('--graph_model', '-g',
                        default='./GCNv2/TrainedGCN/Rat/RatPoseAll_indoor_noise0005_neg07/CP_epoch300.pth',
                        # default='./GCN/TrainedGCN/model1_Vertical_noise0005_neg07/CP_epoch300.pth',
                        # default='./GCN/TrainedGCN/RegModel1_Vertical_noise0005_neg07/CP_epoch300.pth',
                        # default='./GCN/TrainedGCN/3classes/Reg_Vertical_noise0005_neg07/CP_epoch200.pth',
                        metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('-d', '--dir_checkpoint', dest='ckp', type=str,
                        default='./TrainedModels/ContrastiveModels/train_ResSelf_s2_GraphConstrint/training_paralle_GCNv2/',
                        help='Saved model path')
    parser.add_argument("--local_rank", type=int, default=0,
                        help="number of cpu threads to use during batch generation")
    return parser.parse_args()


def train_net(model,
              Graph_model,
              Device,
              dir_checkpoint,
              dir_img,
              source_label,
              target_label,
              target_unlabel,
              epochs=100,
              batch_size=4,
              lr=0.001,
              weight=0.01,
              val_percent=0.1,
              save_cp=True,
              augment=0,
              angle=math.pi):   # scale是输入与输出的边长比

    dataset = DatasetStage2_iteration(resize_w, resize_h, dir_img, source_label, target_label, target_unlabel, num_points, scale=4, angle=angle)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    # 并行
    train_sampler = torch.utils.data.distributed.DistributedSampler(train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=8,
                                               pin_memory=True, sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8,
                                             pin_memory=True, sampler=val_sampler)
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {Device.type}
    ''')

    optimizer = optim.AdamW(net.parameters(), lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)   # 每隔50个epoch降一次学习率（*0.1）

    # criterion = nn.MSELoss()
    # criterion_graph = graph_contrative_loss(Graph_model, num_points, device)
    # criterion_self = self_contrative_loss(Graph_model, num_points, device, angle=angle)
    criterion_self = self_contrative_meanloss(Graph_model, num_points, device, angle=angle)

    val_score_min = 1
    return_loss_all = []
    loss_all = np.zeros([4, epochs])
    for epoch in range(30, epochs):
    # for epoch in range(epochs):
        print('epoch:', epoch)

        model.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                img = batch['image_s']
                img_t = batch['image_t']
                true_heatmaps = batch['heatmap_s']
                true_heatmaps_t = batch['heatmap_t']
                self_supvision = batch['self_supvision']
                self_supvision = np.array(self_supvision)

                img = img.to(device=Device, dtype=torch.float32)
                img_t = img_t.to(device=Device, dtype=torch.float32)
                heatmap_type = torch.float32
                true_heatmaps = true_heatmaps.to(device=Device, dtype=heatmap_type)
                true_heatmaps_t = true_heatmaps_t.to(device=Device, dtype=heatmap_type)

                pred1, pred2 = model(img, img_t)

                loss, return_loss = criterion_self(pred1, pred2, true_heatmaps, true_heatmaps_t, self_supvision)
                temp = np.zeros(4)
                temp[0] = epoch + 1
                temp[1:] = return_loss
                return_loss_all.append(return_loss)

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(img.shape[0])
                global_step += 1
                if global_step % (n_train // (100 * batch_size)) == 0:
                    val_score = eval_net_s2_update(model, GraphNet, num_points, val_loader,
                                                   self_contrative_loss, angle, Device)
                    logging.info('Validation loss: {}'.format(val_score))

        # 保存best模型
        if dist.get_rank() == 0 and val_score < val_score_min:
            val_score_min = val_score
            print("save model. val_score = ", val_score)
            torch.save(model.module.state_dict(), dir_checkpoint + f'CP_best.pth')
            torch.save(model.module.SubResnet.state_dict(), dir_checkpoint + f'Resnet50_best.pth')
            torch.save(model.module.TransformerPart.state_dict(), dir_checkpoint + f'TransformerPart_best.pth')
            torch.save(model.module.PoseHead.state_dict(), dir_checkpoint + f'PoseHead_best.pth')
            print('Best epoch:', epoch + 1)

        scheduler.step()  # 学习率衰减
        print('epoch:', epoch + 1, ' loss:', loss.item())
        loss_all[0, epoch] = epoch + 1
        loss_all[1, epoch] = loss.item()

        # if save_cp:
        # if dist.get_rank() == 0 and (epoch + 1) % epochs == 0:
        if dist.get_rank() == 0 and (epoch + 1) % 3 == 0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(model.module.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            torch.save(model.module.SubResnet.state_dict(),
                       dir_checkpoint + f'Resnet50_epoch{epoch + 1}.pth')
            torch.save(model.module.TransformerPart.state_dict(),
                       dir_checkpoint + f'TransformerPart_epoch{epoch + 1}.pth')
            torch.save(model.module.PoseHead.state_dict(),
                       dir_checkpoint + f'PoseHead_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        # 输出当前学习率
        for param_group in optimizer.param_groups:
            print('Lr of optimizer:', param_group['lr'])

    return loss_all  # , np.array(return_loss_all)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    local_rank = torch.distributed.get_rank()
    print('local_rank:', local_rank)
    torch.cuda.set_device(local_rank)
    global device
    device = torch.device("cuda", local_rank)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    print(args.path_backbone)
    print('input_size:', resize_w, resize_h, ';  Augment:', args.augment)
    print('lr:', args.lr, ';  batch_size:', args.batch)
    print('source:', args.source_label, '; target:', args.target_label)

    isExists = os.path.exists(args.ckp)
    if not isExists:  # 判断结果
        os.makedirs(args.ckp)

    # 图网络
    GraphNet = Net(8, reg=False).to(device=device)
    GraphNet.load_state_dict(torch.load(args.graph_model, map_location=device), strict=False)
    logging.info("Graph model loaded !")

    # 构建网络
    net = ContrastNet1_MultiA(args, extract_list, device, train=True, nof_joints=num_points)
    net.to(device=device)
    # 多卡并行
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,
                                                    find_unused_parameters=True)

    if args.load:
        print(args.load)
        net.load_state_dict(
            torch.load(args.load, map_location=device), strict=False   #strict，该参数默认是True，表示预训练模型的层和自己定义的网络结构层严格对应相等（比如层名和维度）
        )
        logging.info(f'Model loaded from {args.load}')
        print('Pretrained weights have been loaded!')
    else:
        print('No pretrained models have been loaded except the backbone!')
    # print(net)

    # faster convolutions, but more memory
    cudnn.benchmark = True

    pesudo_label_dir = args.target_label
    unlabel_dir = args.target_unlabel
    continue_training = 1

    loss_all = train_net(model=net,
                         Graph_model=GraphNet,
                         Device=device,
                         dir_checkpoint=args.ckp,
                         dir_img=args.Imgs,
                         source_label=args.source_label,
                         target_label=pesudo_label_dir,
                         target_unlabel=unlabel_dir,
                         epochs=args.epochs,
                         batch_size=args.batch,
                         lr=args.lr,
                         weight=args.weight,
                         val_percent=args.val / 100,
                         augment=args.augment,
                         angle=args.angle * math.pi / 180)

    # #  绘制loss曲线
    plt.plot(loss_all[0, :], loss_all[2, :], color='b')  # 绘制mse loss曲线
    plt.xlabel("epochs", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    plt.savefig(args.ckp + 'loss.jpg')
    save_path_loss = args.ckp + 'loss_.npy'
    np.save(save_path_loss, loss_all)
    plt.close()
