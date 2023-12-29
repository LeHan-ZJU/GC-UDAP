import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Function
# from Loss.GraphLoss import graph_contrative_loss

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    heatmap_type = torch.float32# if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion = nn.MSELoss()
    # criterion_2 = Cos_SimlarLoss(int(resize_w / img_scale), int(resize_h / img_scale), relation)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            # imgs = batch['target_img']
            # true_heatmaps = batch['centermap']
            true_heatmaps = batch['heatmap']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_heatmaps = true_heatmaps.to(device=device, dtype=heatmap_type)

            with torch.no_grad():
                heatmaps_pred = net(imgs)

            # tot = criterion(heatmaps_pred, true_heatmaps).item()
            loss_mse = criterion(heatmaps_pred, true_heatmaps).item()
            # loss_similar = loss_weight * criterion_2(heatmaps_pred, true_heatmaps).item()
            tot = loss_mse  # + loss_similar
            # if net.n_classes > 1:
            #     tot = criterion(heatmaps_pred, true_heatmaps).item()
            #     # tot += F.cross_entropy(heatmaps_pred, true_heatmaps).item()
            # else:
            #     pred = torch.sigmoid(heatmaps_pred)
            #     pred = (pred > 0.5).float()
            #     tot += dice_coeff(pred, true_heatmaps).item()
            pbar.update()

    net.train()
    return tot / n_val


def eval_s2(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    heatmap_type = torch.float32# if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion = nn.MSELoss()
    # criterion_2 = Cos_SimlarLoss(int(resize_w / img_scale), int(resize_h / img_scale), relation)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            img = batch['image']
            img_f = batch['image_f']
            true_heatmaps = batch['heatmap']
            true_heatmaps_f = batch['heatmap_f']

            img = img.to(device=device, dtype=torch.float32)
            img_f = img_f.to(device=device, dtype=torch.float32)
            true_heatmaps = true_heatmaps.to(device=device, dtype=heatmap_type)
            true_heatmaps_f = true_heatmaps_f.to(device=device, dtype=heatmap_type)

            pred1, pred2 = net(img, img_f)
            # print('heatmaps:', true_heatmaps.shape, pred1.shape, pred2.shape)

            loss1 = criterion(pred1, true_heatmaps)
            loss2 = criterion(pred2, true_heatmaps_f)
            # loss_graph = criterion_graph(pred1, pred2)
            loss = loss1 + loss2  # + loss_graph

            # loss_similar = loss_weight * criterion_2(heatmaps_pred, true_heatmaps).item()
            tot = loss  # + loss_similar
            # if net.n_classes > 1:
            #     tot = criterion(heatmaps_pred, true_heatmaps).item()
            #     # tot += F.cross_entropy(heatmaps_pred, true_heatmaps).item()
            # else:
            #     pred = torch.sigmoid(heatmaps_pred)
            #     pred = (pred > 0.5).float()
            #     tot += dice_coeff(pred, true_heatmaps).item()
            pbar.update()

    net.train()
    return tot / n_val


def eval_net_mt(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    heatmap_type = torch.float32# if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion = nn.MSELoss()
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_areamap, true_heatmaps = batch['image'], batch['areamap'], batch['heatmap']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_areamap = true_areamap.to(device=device, dtype=heatmap_type)
            true_heatmaps = true_heatmaps.to(device=device, dtype=heatmap_type)

            with torch.no_grad():
                heatmaps_pred, areamap_pred = net(imgs)
                # areamap_pred = net(imgs)

            loss_am = criterion(areamap_pred, true_areamap).item()
            loss_hm = criterion(heatmaps_pred, true_heatmaps).item()
            tot = 0.5 * loss_am + loss_hm
            pbar.update()

    net.train()
    return tot / n_val


def eval_net_contrast(net, loader, Device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion = nn.MSELoss()
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            img = batch['image_s']
            img_t = batch['image_t']
            true_heatmaps = batch['heatmap_s']
            true_heatmaps_t = batch['heatmap_t']

            img = img.to(device=Device, dtype=torch.float32)
            img_t = img_t.to(device=Device, dtype=torch.float32)
            heatmap_type = torch.float32
            true_heatmaps = true_heatmaps.to(device=Device, dtype=heatmap_type)
            true_heatmaps_t = true_heatmaps_t.to(device=Device, dtype=heatmap_type)

            with torch.no_grad():
                pred1, pred2 = net(img, img_t)

                # print('area:', true_areamap.shape, areamap_pred.shape)
                # print('hm:', true_heatmaps.shape, heatmaps_pred.shape)

            loss1 = criterion(pred1, true_heatmaps).item()
            loss2 = criterion(pred2, true_heatmaps_t).item()
            tot = loss1 + loss2
            pbar.update()

    net.train()
    return tot / n_val


def eval_net_s2_update(net, Graph_model, num_points, loader, self_contrative_loss, angle, Device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
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


def eval_contrast_center(net, loader, device, whl, wal):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    heatmap_type = torch.float32# if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion = nn.MSELoss()
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs1 = batch['image']
            imgs2 = batch['target_img']
            true_heatmap = batch['heatmap']
            true_centermap = batch['centermap']

            imgs1 = imgs1.to(device=device, dtype=torch.float32)
            imgs2 = imgs2.to(device=device, dtype=torch.float32)
            true_heatmap = true_heatmap.to(device=device, dtype=torch.float32)
            true_centermap = true_centermap.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                heatmaps_pred, centermap_pred = net(imgs1, imgs2)
                # print('area:', true_areamap.shape, areamap_pred.shape)
                # print('hm:', true_heatmaps.shape, heatmaps_pred.shape)

            loss_cm = criterion(centermap_pred, true_centermap).item()
            loss_hm = criterion(heatmaps_pred, true_heatmap).item()
            tot = wal * loss_cm + whl * loss_hm
            pbar.update()

    net.train()
    return tot / n_val


def eval_2hms(net, loader, device, whl, wal):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    heatmap_type = torch.float32# if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion = nn.MSELoss()
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs1 = batch['img_1']
            imgs2 = batch['img_2']
            true_heatmaps1 = batch['heatmaps_1']
            true_heatmaps2 = batch['heatmaps_2']

            imgs1 = imgs1.to(device=device, dtype=torch.float32)
            imgs2 = imgs2.to(device=device, dtype=torch.float32)
            true_heatmaps1 = true_heatmaps1.to(device=device, dtype=torch.float32)
            true_heatmaps2 = true_heatmaps2.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                heatmaps1_pred, heatmaps2_pred = net(imgs1, imgs2)
                # print('area:', true_centermap.shape, centermap_pred.shape)
                # print('hm:', true_heatmaps.shape, heatmaps_pred.shape)

            loss_hm1 = criterion(heatmaps1_pred, true_heatmaps1)
            loss_hm2 = criterion(heatmaps2_pred, true_heatmaps2)

            tot = wal * loss_hm1 + whl * loss_hm2
            pbar.update()

    net.train()
    return tot / n_val


def eval_contrast_reg(net, loader, device, whl, wal):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    heatmap_type = torch.float32# if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion = nn.MSELoss()
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            source_imgs, target_imgs, true_points, true_center = \
                batch['source_img'], batch['target_img'], batch['points_all'], batch['center']
            source_imgs = source_imgs.to(device=device, dtype=torch.float32)
            target_imgs = target_imgs.to(device=device, dtype=torch.float32)
            true_points = true_points.to(device=device, dtype=torch.float32)
            true_center = true_center.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                points_pred, center_pred = net(source_imgs, target_imgs)
                # print('area:', true_areamap.shape, areamap_pred.shape)
                # print('hm:', true_heatmaps.shape, heatmaps_pred.shape)

            loss_am = criterion(center_pred, true_center)
            loss_hm = criterion(points_pred, true_points)
            tot = wal * loss_am + whl * loss_hm
            pbar.update()

    net.train()
    return tot / n_val


def eval_center(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    heatmap_type = torch.float32# if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion = nn.MSELoss()
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            source_imgs, target_imgs, true_centermap_s, true_centermap_t = \
                batch['source_img'], batch['target_img'], batch['centermap_s'], batch['centermap_t']
            source_imgs = source_imgs.to(device=device, dtype=torch.float32)
            target_imgs = target_imgs.to(device=device, dtype=torch.float32)
            true_centermap_s = true_centermap_s.to(device=device, dtype=torch.float32)
            true_centermap_t = true_centermap_t.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                source_pred, target_pred = net(source_imgs, target_imgs)
                # print('area:', true_areamap.shape, areamap_pred.shape)
                # print('hm:', true_heatmaps.shape, heatmaps_pred.shape)

            loss_s = criterion(source_pred, true_centermap_s).item()
            loss_t = criterion(target_pred, true_centermap_t).item()
            tot = loss_s + loss_t
            pbar.update()

    net.train()
    return tot / n_val
