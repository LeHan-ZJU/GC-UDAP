import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from Models.eval_pose import eval_net
from Models.ContrastModel import ResSelf_pre

from utils.dataset_csv import DatasetPoseCSV
from torch.utils.data import DataLoader, random_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

num_points = 6

resize_w = 640
resize_h = 480
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
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch', metavar='B', dest='batch', type=int, nargs='?', default=1, help='Batch size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str,  default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=4, help='the ratio between img and GT')
    parser.add_argument('-w', '--weight', dest='weight', type=float, default=0, help='the weight of similar loss')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=5.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-t', '--dir_label', dest='dir_label', type=str, default='',
                        help='Label path of source image label')
    parser.add_argument('-i', '--imgs', dest='Imgs', type=str, default='',
                        help='File path of images')
    parser.add_argument('-p', '--path_backbone', dest='path_backbone', type=str, default=None,
                        help='the path of backbone')
    parser.add_argument('-d', '--dir_checkpoint', dest='ckp', type=str, default='./TrainedModels/',
                        help='Saved model path')
    return parser.parse_args()


def train_net(model,
              Device,
              dir_checkpoint,
              dir_img,
              dir_label,
              epochs=30,
              batch_size=4,
              lr=0.001,
              weight=0.01,
              val_percent=0.1,
              save_cp=True,
              img_scale=1,
              augment=0):

    dataset = DatasetPoseCSV(resize_w, resize_h, dir_img, dir_label, img_scale, num_points)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {Device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    criterion = nn.MSELoss()
    val_score_min = 1
    loss_all = np.zeros([4, epochs])
    for epoch in range(epochs):

        model.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_heatmaps = batch['heatmap']

                imgs = imgs.to(device=Device, dtype=torch.float32)
                heatmap_type = torch.float32 
                true_heatmaps = true_heatmaps.to(device=Device, dtype=heatmap_type)

                heatmaps_pred = model(imgs)

                loss = criterion(heatmaps_pred, true_heatmaps)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:

                    val_score = eval_net(model, val_loader, Device)
                    logging.info('Validation loss: {}'.format(val_score))

                    if val_score < val_score_min:
                        val_score_min = val_score
                        print("save model. val_score = ", val_score)
                        torch.save(model.state_dict(),
                                   dir_checkpoint + f'CP_best_epoch{epoch + 1}.pth')
                        torch.save(model.SubResnet.state_dict(),
                                   dir_checkpoint + f'Backbone_best_epoch{epoch + 1}.pth')
                        torch.save(model.TransformerPart.state_dict(),
                                   dir_checkpoint + f'TransformerPart_best_epoch{epoch + 1}.pth')
                        torch.save(model.PoseHead.state_dict(),
                                   dir_checkpoint + f'PoseHead_best_epoch{epoch + 1}.pth')

        scheduler.step()
        print('epoch:', epoch + 1, ' loss:', loss.item())
        loss_all[0, epoch] = epoch + 1
        loss_all[1, epoch] = loss.item()

        if (epoch + 1) % 100 == 0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(model.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            torch.save(model.SubResnet.state_dict(),
                       dir_checkpoint + f'Resnet50_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        for param_group in optimizer.param_groups:
            print('Lr of optimizer:', param_group['lr'])

    return loss_all


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    print(args.path_backbone)
    print('input_size:', resize_w, resize_h, ';  Augment:', args.augment)
    print('lr:', args.lr, ';  batch_size:', args.batch, ';  the weight of similar loss:', args.weight)
    print('trainset:', args.dir_label)

    isExists = os.path.exists(args.ckp)
    if not isExists:
        os.makedirs(args.ckp)

    net = ResSelf_pre(args, extract_list, device, train=True, nof_joints=num_points)

    if args.load:
        print(args.load)
        net.load_state_dict(
            torch.load(args.load, map_location=device), strict=False
        )
        logging.info(f'Model loaded from {args.load}')
        print('Pretrained weights have been loaded!')
    else:
        print('No pretrained models have been loaded except the backbone!')

    logging.info(f'Network:\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    net.to(device=device)

    try:
        loss_all = train_net(model=net,
                             Device=device,
                             dir_checkpoint=args.ckp,
                             dir_img=args.Imgs,
                             dir_label=args.dir_label,
                             epochs=args.epochs,
                             batch_size=args.batch,
                             lr=args.lr,
                             weight=args.weight,
                             val_percent=args.val / 100,
                             img_scale=args.scale,
                             augment=args.augment)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

