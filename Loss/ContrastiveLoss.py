import cv2
import math
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn import functional as F

from Tools.process import rotate_img
from Loss.GraphLoss import GraphContrastiveLoss


def rotate_heatmap_3d(heatmap, angle):
    B, _, _, _ = heatmap.shape
    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), 0],
        [math.sin(angle), math.cos(angle), 0]
    ], dtype=torch.float)
    transform_matrix = theta.unsqueeze(0).repeat(B, 1, 1)
    grid = F.affine_grid(transform_matrix, heatmap.size())
    output = F.grid_sample(heatmap, grid)
    return output


def rotate_heatmap(heatmap, angle, device):
    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), 0],
        [math.sin(angle), math.cos(angle), 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), heatmap.unsqueeze(0).size())
    output = F.grid_sample(heatmap.unsqueeze(0), grid.to(device))
    return output[0]


class self_contrative_loss(torch.nn.Module):
    def __init__(self, GraphNet, num_points, device, angle):
        super(self_contrative_loss, self).__init__()
        self.GraphNet = GraphNet
        self.num_points = num_points
        self.device = device
        self.angle = angle
        self.mse = torch.nn.MSELoss()
        self.GraphContrastiveLoss = GraphContrastiveLoss(GraphNet, num_points, device)

    def forward(self, heatmaps1, heatmaps2):
        loss = 0
        B = heatmaps1.shape[0]
        for i in range(B):
            hm2 = rotate_heatmap(heatmaps2[i, :, :, :], self.angle, self.device)
            # plt.subplot(2, 2, 1)
            # plt.imshow(heatmaps2[i, 0:3, :, :].detach().cpu().numpy().transpose(1, 2, 0))
            # plt.subplot(2, 2, 2)
            # plt.imshow(hm2[0:3, :, :].detach().cpu().numpy().transpose(1, 2, 0))
            # plt.subplot(2, 2, 3)
            # plt.imshow(heatmaps1[i, 0:3, :, :].detach().cpu().numpy().transpose(1, 2, 0))
            # plt.show()

            GraphScore = self.GraphContrastiveLoss(heatmaps1[i, :, :, :], hm2)
            l2 = self.mse(heatmaps1[i, :, :, :], heatmaps2[i, :, :, :])
            loss = loss + torch.mul(l2, torch.exp(-GraphScore))
        return loss / B


class self_contrative_meanloss(torch.nn.Module):
    def __init__(self, GraphNet, num_points, device, angle):
        super(self_contrative_meanloss, self).__init__()
        self.GraphNet = GraphNet
        self.num_points = num_points
        self.device = device
        self.angle = angle
        self.mse = torch.nn.MSELoss()
        self.GraphContrastiveLoss = GraphContrastiveLoss(GraphNet, num_points, device)

    def forward(self, pred1, pred2, gt1, gt2, self_supvision):
        loss = 0
        B = pred1.shape[0]
        return_loss = np.zeros(3)
        for i in range(B):
            if self_supvision[i][0] == 1:
                hm2 = rotate_heatmap(pred2[i, :, :, :], self.angle, self.device)
                GraphScore = self.GraphContrastiveLoss(pred1[i, :, :, :], hm2)
                l2 = self.mse(pred1[i, :, :, :], pred2[i, :, :, :])
                self_loss = torch.mul(l2, torch.exp(-GraphScore))
                loss = loss + self_loss
                return_loss[0] = self_loss.item()
            else:
                loss1 = self.mse(pred1, gt1)
                loss2 = self.mse(pred2, gt2)
                loss = loss + loss1 + loss2
                return_loss[1:] = [loss1.item(), loss2.item()]
        return loss / B, return_loss


def debug_rotate_heatmap(heatmap, angle):
    B, _, _, _ = heatmap.shape
    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), 0],
        [math.sin(angle), math.cos(angle), 0]
    ], dtype=torch.float)
    transform_matrix = theta.unsqueeze(0).repeat(B, 1, 1)
    grid = F.affine_grid(transform_matrix, heatmap.size())
    output = F.grid_sample(heatmap, grid)
    print(output.shape)
    new_img_torch1 = output[0]
    new_img_torch2 = output[1]

    # grid = F.affine_grid(theta.unsqueeze(0), heatmap.unsqueeze(0).size())
    # output = F.grid_sample(heatmap.unsqueeze(0), grid)
    # new_img_torch = output[0]
    # plt.imshow(new_img_torch.numpy().transpose(1, 2, 0))
    # plt.show()
    return new_img_torch1.numpy().transpose(1, 2, 0), new_img_torch2.numpy().transpose(1, 2, 0)


if __name__ == "__main__":
    device = torch.device('cpu')
    img_path = "G:/Data/RatPose/all/Vertical_Indoor/4_0.jpg"
    img_path2 = "G:/Data/RatPose/all/Vertical_Indoor/20211016_2-1_2_1280.jpg"
    A = Image.open(img_path)
    B = cv2.imread(img_path)
    print(B.shape)
    Br = rotate_img(B, 180, [B.shape[1]/2, B.shape[0]/2])
    b, g, r = cv2.split(Br)
    Br = cv2.merge([r, g, b])

    img_torch0 = transforms.ToTensor()(B)

    img_torch = transforms.ToTensor()(Image.open(img_path))
    img_torch2 = transforms.ToTensor()(Image.open(img_path2))
    print(img_torch.shape)

    # hms = torch.cat([img_torch.unsqueeze(0), img_torch2.unsqueeze(0)], dim=0)
    hms = img_torch
    print(hms.shape)

    angle = -180 * math.pi / 180

    # rot_img1, rot_img2 = debug_rotate_heatmap(hms, angle)
    rot_img1 = rotate_heatmap(img_torch, angle, device)
    print(img_torch.shape, rot_img1.shape)
    rot_img1 = rot_img1.numpy().transpose(1, 2, 0)


    plt.subplot(2, 2, 1)
    plt.imshow(img_torch.numpy().transpose(1, 2, 0))
    plt.subplot(2, 2, 2)
    plt.imshow(Br-rot_img1)

    plt.subplot(2, 2, 3)
    plt.imshow(rot_img1)
    plt.subplot(2, 2, 4)
    plt.imshow(Br)
    plt.show()

