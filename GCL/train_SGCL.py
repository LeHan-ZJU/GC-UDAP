import os
import logging
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
from GCL.gcn import Net
from GCL.dataset_gcn_animal import gcn_dataset_rat


def get_args():
    parser = argparse.ArgumentParser(description='Train the SGCL module',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-n', '--num_points', metavar='N', type=int, default=10,
                        help='Number of keypoints', dest='num_points')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=64,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('-t', '--label', dest='label', type=str, default='./label_path', help='Label path')
    parser.add_argument('-i', '--imgs', dest='Imgs', type=str, default='./img_path/',
                        help='File path of images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-d', '--dir_checkpoint', dest='ckp', type=str, default='./TrainedSGCL/',
                        help='Saved model path')
    parser.add_argument('-pr', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    return parser.parse_args()


def train(loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output.view(-1), data.y.float())
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    scheduler.step()
    return total_loss / len(dataset)


def validation(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = (output > 0.5).float()
        correct += (pred == data.y).float().sum()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    isExists = os.path.exists(args.ckp)
    if not isExists:
        os.makedirs(args.ckp)

    dataset = gcn_dataset_rat(args.label, args.Imgs, args.num_points, noise_p=0.0005, noise_n=0.2)
    train_dataset = dataset[:int(len(dataset) * (1 - args.val / 100))]
    val_dataset = dataset[int(len(dataset) * (1 - args.val / 100)):]
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset[0].num_node_features, reg=0).to(device)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device), strict=False)
        logging.info("Graph model loaded !")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = torch.nn.BCELoss()

    loss_all = np.zeros([3, args.epochs])
    acc_all = np.zeros([2, int(args.epochs / 10)])
    for epoch in range(1, args.epochs + 1):
        loss = train(train_loader)
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
        loss_all[0, epoch - 1] = epoch
        loss_all[1, epoch - 1] = loss

        if epoch % 10 == 0:
            val_acc = validation(val_loader)
            acc_all[0, int(epoch / 10) - 1] = epoch
            acc_all[1, int(epoch / 10) - 1] = val_acc
            loss_all[2, epoch - 1] = val_acc

    torch.save(model.state_dict(),
               args.ckp + f'CP_epoch{args.epochs}.pth')
    logging.info(f'Checkpoint {args.epochs} saved !')
