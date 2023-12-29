import os
import logging
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
from GCL.gcn import Net, Net_l
from GCL.dataset_gcn_animal import gcn_dataset_rat


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target heatmaps',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-n', '--num_points', metavar='N', type=int, default=19,
                        help='Number of keypoints', dest='num_points')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=64,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('-t', '--label', dest='label', type=str,
                        default='D:/Data/Pose/TigDog/behaviorDiscovery2.0/landmarks/horse/horse0.csv',
                        # default='D:/Data/Pose/TigDog/behaviorDiscovery2.0/landmarks/horse_tiger.csv/',
                        # default='G:/Data/RatPose/label/RatPoseLabels/VerticalTilt_crop_v2.csv',
                        # default='G:/Data/RatPose/label/RatPoseLabels/RatPoseAll_indoor_crop_v2_trainval.csv',
                        help='Label path')
    parser.add_argument('-i', '--imgs', dest='Imgs', type=str,
                        default='D:/Data/Pose/TigDog/behaviorDiscovery2.0/',  # TigDog
                        # default='G:/Data/RatPose/all/',
                        help='File path of images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-d', '--dir_checkpoint', dest='ckp', type=str,
                        default='./TrainedGCN/TigDog/4layers_horse_noise001_neg06/continue/',
                        # default='./TrainedGCN/Rat/RatPoseAll_indoor_noise0005_neg07_2neighbor/',
                        # default='./TrainedGCN/Rat/VerticalTilt_indoor_noise0005_neg07/',
                        help='Saved model path')
    parser.add_argument('-pr', '--load', dest='load', type=str,
                        # default=False,
                        default='./TrainedGCN/TigDog/4layers_horse_noise001_neg06/CP_epoch300.pth',
                        # default='./TrainedGCN/TigDog/horse_noise0005_neg07/CP_epoch300.pth',
                        help='Load model from a .pth file')
    return parser.parse_args()


def train(loader):
    model.train()
    total_loss = 0
    # print(len(loader))
    # for i in range(len(loader)):
    for data in loader:
        # data = loader[i]
        data = data.to(device)
        optimizer.zero_grad()
        # print('x:', data.x)
        output = model(data.x, data.edge_index, data.batch)
        # print('out:', output.view(-1), data.y.float())
        loss = criterion(output.view(-1), data.y.float())
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    scheduler.step()  # 学习率衰减
    return total_loss / len(dataset)


def validation(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = (output > 0.5).float()
        correct += (pred == data.y).float().sum()
        # print(int(np.array(pred.item())), np.array(data.y.item()))
    return correct / len(loader.dataset)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    isExists = os.path.exists(args.ckp)
    if not isExists:  # 判断保存路径是否存在
        os.makedirs(args.ckp)

    # dataset = gcn_dataset_3classes(args.label, args.Imgs, args.num_points, noise_p=0.0001, noise_n=0.5)  # rat
    # dataset = gcn_dataset_rat(args.label, args.Imgs, args.num_points, noise_p=0.0005, noise_n=0.7)  # rat
    dataset = gcn_dataset_rat(args.label, args.Imgs, args.num_points, noise_p=0.001, noise_n=0.6)  # tigdog
    print('dataset:', len(dataset), 'trainset:', int(len(dataset) * (1 - args.val / 100)),
          'num_nodes:', dataset[0].num_node_features)
    train_dataset = dataset[:int(len(dataset) * (1 - args.val / 100))]
    val_dataset = dataset[int(len(dataset) * (1 - args.val / 100)):]
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # dataset = TUDataset(root='./datasets', name='PROTEINS')
    # loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('num_node_features：', dataset[0].num_node_features)
    # model = Net(dataset[0].num_node_features, reg=0).to(device)
    model = Net_l(dataset[0].num_node_features, reg=0).to(device)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device), strict=False)
        logging.info("Graph model loaded !")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 每隔50个epoch降一次学习率（*0.1）
    criterion = torch.nn.BCELoss()
    # criterion = torch.nn.MSELoss()

    loss_all = np.zeros([3, args.epochs])
    acc_all = np.zeros([2, int(args.epochs / 10)])
    for epoch in range(1, args.epochs + 1):
        loss = train(train_loader)
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
        loss_all[0, epoch - 1] = epoch
        loss_all[1, epoch - 1] = loss

        if epoch % 10 == 0:
            val_acc = validation(val_loader)
            # print('Test Accuracy: {:.4f}'.format(test_acc))
            acc_all[0, int(epoch / 10) - 1] = epoch
            acc_all[1, int(epoch / 10) - 1] = val_acc
            loss_all[2, epoch - 1] = val_acc

            # 输出当前学习率
            for param_group in optimizer.param_groups:
                print('Lr of optimizer:', param_group['lr'], '  Validation Accuracy: {:.4f}'.format(val_acc))

    torch.save(model.state_dict(),
               args.ckp + f'CP_epoch{args.epochs}.pth')
    logging.info(f'Checkpoint {args.epochs} saved !')

    # 绘制loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss_all[0, :], loss_all[1, :], color='b')  # 绘制source_loss曲线
    plt.xlabel("epochs", fontsize=12)
    plt.ylabel("loss", fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(acc_all[0, :], acc_all[1, :], color='b')  # 绘制target_loss曲线
    plt.xlabel("epochs", fontsize=12)
    plt.ylabel("test_acc", fontsize=12)

    plt.savefig(args.ckp + 'loss_acc.jpg')
    save_path_loss = args.ckp + 'loss_acc.npy'
    np.save(save_path_loss, loss_all)
