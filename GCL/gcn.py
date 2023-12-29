import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import torch
from torch_geometric.nn.conv import GCNConv
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

    def __init__(self, num_node_features):
        super(Net_S, self).__init__()
        self.conv1_1 = GINConvModule(num_node_features)
        self.conv2_1 = GINConvModule(64)
        self.conv1_2 = GINConvModule(num_node_features)
        self.conv2_2 = GINConvModule(64)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x1, x2, edge_index, batch):
        x1 = F.relu(self.conv1_1(x1, edge_index))
        x1 = F.relu(self.conv2_1(x1, edge_index))

        x2 = F.relu(self.conv1_2(x2, edge_index))
        x2 = F.relu(self.conv2_2(x2, edge_index))
        x = torch.cat((x1, x2), dim=0)
        x = global_add_pool(x, batch)
        x = self.fc(x)
        return torch.sigmoid(x)


    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data1, data2):
        x1, edge_index1 = data1.x, data1.edge_index
        x2, edge_index2 = data2.x, data2.edge_index

        x1 = self.conv1(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, training=self.training)
        x1 = self.conv2(x1, edge_index1)

        x2 = self.conv1(x2, edge_index1)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, training=self.training)
        x2 = self.conv2(x2, edge_index1)

        return F.log_softmax(x1, dim=1)