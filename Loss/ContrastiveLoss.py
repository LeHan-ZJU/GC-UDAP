import cv2
import math
import torch
import numpy as np
from torch.nn import functional as F
from Loss.GraphLoss import GraphContrastiveLoss


def rotate_heatmap(heatmap, angle, device):
    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), 0],
        [math.sin(angle), math.cos(angle), 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), heatmap.unsqueeze(0).size())
    output = F.grid_sample(heatmap.unsqueeze(0), grid.to(device))
    return output[0]


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
        for i in range(B):
            # If samples are from a target subset with pseudo-labels and the source dataset, the mean squared error (MSE) is computed.
            if self_supvision[i][0] == 1:     
                hm2 = rotate_heatmap(pred2[i, :, :, :], self.angle, self.device)
                GraphScore = self.GraphContrastiveLoss(pred1[i, :, :, :], hm2)
                l2 = self.mse(pred1[i, :, :, :], pred2[i, :, :, :])
                self_loss = torch.mul(l2, torch.exp(-GraphScore))
                loss = loss + self_loss
                            
            # If samples are from a target subset without pseudo-labels, training is guided by computing graph constraints.
            else:
                loss1 = self.mse(pred1, gt1)
                loss2 = self.mse(pred2, gt2)
                loss = loss + loss1 + loss2
                
        return loss / B
