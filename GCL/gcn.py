import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class GINConvModule(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.GINConvModule = GINConv(torch.nn.Sequential(
            torch.nn.Linear(num_node_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64)))

    def forward(self, x, edge_index):
        return self.GINConvModule(x, edge_index)


class Net(torch.nn.Module):
    def __init__(self, num_node_features, reg):
        super(Net, self).__init__()
        self.reg = reg
        self.conv1 = GINConvModule(num_node_features)
        self.conv2 = GINConvModule(64)
        self.conv3 = GINConvModule(64)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_add_pool(x, batch)
        x = self.fc(x)
        if self.reg:
            return x
        else:
            return torch.sigmoid(x)
