import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from SpecialSP import SpecialSpmm
from attention import Attention
from mlp import MLP


class GNNLayer(nn.Module):
    def __init__(self, num_mlp_layers, input_dim, hidden_dim, output_dim, num_heads):
        super(GNNLayer, self).__init__()
        self.num_heads = num_heads
        self.special_spmm = SpecialSpmm()
        self.mlp_es = torch.nn.ModuleList()
        self.edge_wt = torch.nn.ModuleList()

        for heads in range(num_heads):
            self.mlp_es.append(MLP(num_mlp_layers, input_dim, hidden_dim, output_dim))
            self.edge_wt.append(Attention(output_dim * 2 + 1))

    def forward(self, x, Adj_block):
        idx, elem, shape = Adj_block
        h = []
        num_r = x.shape[0]
        print(shape)
        for heads in range(self.num_heads):
            features = self.mlp_es[heads](x)
            features = [features[idx[0]], features[idx[1]], elem.unsqueeze(1)]
            features = torch.cat(features, dim=1)
            elem_new = torch.exp(-F.leaky_relu(self.edge_wt[heads](features) / 20).squeeze(1))
            assert not torch.isnan(elem_new).any()
            print(features.shape)
            pooled = self.special_spmm(idx, elem_new, shape, features)
            row_sum = self.special_spmm(idx, elem_new, shape, torch.ones(size=(num_r, 1), device=self.device))
            pooled = pooled.div(row_sum)
            h.append(pooled)
        h = torch.cat(h, dim=1)
        return h
