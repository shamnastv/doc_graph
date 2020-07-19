import torch
import utils as u
from argparse import Namespace
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn as nn
import math


class Sp_GCN(torch.nn.Module):
    def __init__(self, args, activation):
        super().__init__()
        self.activation = activation
        self.num_layers = args.num_layers

        self.w_list = nn.ParameterList()
        for i in range(self.num_layers):
            w_i = Parameter(torch.Tensor(args.layer[i], args.layer[i + 1]))
            u.reset_param(w_i)
            self.w_list.append(w_i)

    def forward(self, adj, feats, nodes_mask_list):
        new_feat = self.activation(adj.matmul(feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            new_feat = self.activation(adj.matmul(new_feat.matmul(self.w_list[i])))
        return new_feat

