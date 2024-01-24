import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
# from Loss.GraphLoss import graph_contrative_loss



def eval_net(net, loader, device):
    """Evaluation"""
    net.eval()
    heatmap_type = torch.float32
    n_val = len(loader)
    tot = 0

    criterion = nn.MSELoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_heatmaps = batch['heatmap']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_heatmaps = true_heatmaps.to(device=device, dtype=heatmap_type)

            with torch.no_grad():
                heatmaps_pred = net(imgs)

            loss_mse = criterion(heatmaps_pred, true_heatmaps).item()
            tot = loss_mse
            pbar.update()

    net.train()
    return tot / n_val


def eval_net_s2_update(net, Graph_model, num_points, loader, self_contrative_loss, angle, Device):
    """Evaluation"""
    net.eval()
    n_val = len(loader)
    tot = 0

    criterion = nn.MSELoss()
    criterion_self = self_contrative_loss(Graph_model, num_points, Device, angle=angle)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            img = batch['image_s']
            img_t = batch['image_t']
            true_heatmaps = batch['heatmap_s']
            true_heatmaps_t = batch['heatmap_t']
            self_supvision = batch['self_supvision']
            self_supvision = np.array(self_supvision[0])

            img = img.to(device=Device, dtype=torch.float32)
            img_t = img_t.to(device=Device, dtype=torch.float32)
            heatmap_type = torch.float32
            true_heatmaps = true_heatmaps.to(device=Device, dtype=heatmap_type)
            true_heatmaps_t = true_heatmaps_t.to(device=Device, dtype=heatmap_type)

            with torch.no_grad():
                pred1, pred2 = net(img, img_t)

            if self_supvision[0] == 0:
                loss1 = criterion(pred1, true_heatmaps)
                loss2 = criterion(pred2, true_heatmaps_t)
                loss = loss1 + loss2

                tot += loss.item()
                pbar.update()
            else:
                loss = criterion_self(pred1, pred2)
                tot += loss.item()
                pbar.update()
    net.train()
    return tot / n_val
