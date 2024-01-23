import torch
import numpy as np
import torch.nn.functional as F

from GCN.dataset_gcn_animal import read_edge_index, Angle_relation_multiAngles, softargmax2d
from torch_geometric.data import Data


def graph_inference(GNN, points1, points2, num_points, device):
    edge_index = read_edge_index(num_points)
    x = np.ones([8, num_points * 2])
    x = torch.tensor(x)
    x.to(device)
    x[0:2, 0:num_points] = points1.T
    x[0:2, num_points:] = points2.T

    angles1 = Angle_relation_multiAngles(x[0:2, 0:num_points], edge_index.T, num_points)
    angles2 = Angle_relation_multiAngles(x[0:2, num_points:], edge_index.T, num_points)
    x[2:, 0:num_points] = angles1
    x[2:, num_points:] = angles2

    x = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x.t(), edge_index=edge_index.t())

    GNN.eval()
    data = data.to(device)
    output = GNN(data.x, data.edge_index, data.batch)
    pred = output.float()

    return pred



class GraphContrastiveLoss(torch.nn.Module):
    def __init__(self, GraphNet, num_points, device):
        super(GraphContrastiveLoss, self).__init__()
        self.GraphNet = GraphNet
        self.num_points = num_points
        self.device = device

    def forward(self, heatmaps1, heatmaps2):
        points1 = softargmax2d(heatmaps1, self.device)
        points2 = softargmax2d(heatmaps2, self.device)

        # Graph Inference
        return graph_inference(self.GraphNet, points1, points2, self.num_points, self.device)
