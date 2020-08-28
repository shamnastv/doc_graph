import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp import MLP


class ClusterNN(nn.Module):
    def __init__(self, num_class, dim, num_mlp_layers):
        super(ClusterNN, self).__init__()
        self.centroids = nn.Parameter(torch.zeros(num_class, dim))
        self.mlp_c = MLP(num_mlp_layers, dim, dim, num_class)
        self.batch_norm_c = nn.BatchNorm1d(num_class)

    def forward(self, ge):
        cg = self.mlp_c(ge)
        cg = self.batch_norm_c(cg)
        cg = F.softmax(cg, dim=0)
        return cg
