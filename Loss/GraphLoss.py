import torch
import numpy as np
import torch.nn.functional as F

from GCN.gcn import Net
from GCN.dataset_gcn_animal import read_edge_index
from torch_geometric.data import Data


def find_neighbor(sort, edge_index):
    neighbor = []
    size_edge = edge_index.shape
    for n in range(size_edge[1]):
        if edge_index[0, n] == sort:
            neighbor.append(edge_index[1, n])
    return np.array(neighbor)


def cal_Angle(v1, v2):
    # 计算两个向量间的夹角
    # norm1 = np.linalg.norm(v1)
    # norm2 = np.linalg.norm(v2)
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    # m = np.dot(v1, v2)
    dot_product = torch.dot(v1, v2)
    # cos_ = m / (norm1 * norm2)
    # inv = np.arccos(cos_) * 180 / np.pi
    angle_rad = torch.acos(dot_product / (norm1 * norm2))
    # angle_deg = torch.degrees(angle_rad)
    angle_deg = angle_rad * 180 / np.pi

    if torch.isnan(angle_deg).any():
        angle_deg = 0

    return angle_deg


def Angle_relation_multiAngles(AllPoints, edge_index, num_points):
    angles = []
    for p in range(num_points):
        if num_points == 6:
            angle = np.ones(1)
        else:
            angle = np.ones(6)
        point0 = AllPoints[0:2, p]
        if min(point0) > 0:
            # 根据edge_index寻找与它相邻的两个点
            neighbors = find_neighbor(p, edge_index)
            # print('neighbors:', p, neighbors, len(neighbors))
            if len(neighbors) >= 2:
                s = 0
                for v1 in range(len(neighbors) - 1):
                    for v2 in range(v1 + 1, len(neighbors)):
                        if max(AllPoints[0:2, neighbors[v1]]) == 0 and min(AllPoints[0:2, neighbors[v2]]) > 0:
                            angle[s] = 0  # p非空时，v1和v2任一为[0, 0] --> a=0
                        elif max(AllPoints[0:2, neighbors[v2]]) == 0 and min(AllPoints[0:2, neighbors[v1]]) > 0:
                            angle[s] = 0
                        elif max(AllPoints[0:2, neighbors[v1]]) == 0 and max(AllPoints[0:2, neighbors[v2]]) == 0:
                            angle[s] = -1    # p非空时，v1和v2均为[0, 0] --> a=-1
                        else:
                            # if min(AllPoints[:, neighbors[v1]]) > 0 and min(AllPoints[:, neighbors[v2]]) > 0:
                            vector1 = AllPoints[0:2, neighbors[v1]] - point0
                            vector2 = AllPoints[0:2, neighbors[v2]] - point0
                            angle[s] = cal_Angle(vector1, vector2)
                        s = s + 1
        else:
            angle = -2 * angle  # p:[0, 0] --> a=-2
        angle = np.array(angle)
        # angle = angle[0]

        angles.append(angle)
    angles = np.array(angles)
    # angles = angles.astype(np.float32)

    return torch.tensor(angles)


def softargmax2d(input, device, beta=100):
    *_, h, w = input.shape

    input = input.reshape(*_, h * w)
    input = F.softmax(beta * input, dim=-1)

    indices_c, indices_r = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h), indexing='xy')

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w)))
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w)))
    indices_r = indices_r.to(device)
    indices_c = indices_c.to(device)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result


def graph_inference(GNN, points1, points2, num_points, device):
    edge_index = read_edge_index(num_points)
    x = np.ones([8, num_points * 2])
    x = torch.tensor(x)
    x.to(device)
    # print('points1:', points1.shape, points1)
    x[0:2, 0:num_points] = points1.T
    x[0:2, num_points:] = points2.T
    # x0 = torch.cat((points1.T, points2.T), 1)
    # print(points1.T.shape, points1.T, x, x0, x.shape, x0.shape)

    # 计算夹角
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
    # pred = (output > 0.5).float()
    pred = output.float()
    # print(output, pred)

    # return np.array(pred.item())
    return pred


def GraphContrastiveLoss000(heatmaps1, heatmaps2, GraphNet, num_points, device):

    # heatmaps --> coordinates
    points1 = softargmax2d(heatmaps1, device)
    points2 = softargmax2d(heatmaps2, device)
    # print('p1:', points1[0], points2[0])

    # Graph Inference
    ContrastiveScore = graph_inference(GraphNet, points1[0], points2[0], num_points, device)
    # print('Score:', ContrastiveScore)

    return ContrastiveScore


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


class graph_contrative_loss(torch.nn.Module):
    def __init__(self, GraphNet, num_points, device):
        super(graph_contrative_loss, self).__init__()
        self.GraphContrastiveLoss = GraphContrastiveLoss(GraphNet, num_points, device)

    def forward(self, p1, p2):
        loss = 0
        B = p1.shape[0]
        for i in range(B):
            loss = loss + 1 - self.GraphContrastiveLoss(p1[i, :, :, :], p2[i, :, :, :])
            # print('loss', loss, ', exp:', torch.exp(-loss))
        return loss / B


if __name__ == "__main__":
    GraphModel_path = './TrainedModels/ContrastiveModels/pretrain_ResSelf/debug1/CP_epoch300.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 图网络
    GraphNet = Net(8).to(device=device)
    GraphNet.load_state_dict(torch.load(GraphModel_path, map_location=device), strict=False)
