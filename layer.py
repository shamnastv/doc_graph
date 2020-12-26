import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from SpecialSP import SpecialSpmm
from attention import Attention
from mlp import MLP


class GNNLayer(nn.Module):
    def __init__(self, num_mlp_layers, input_dim, hidden_dim, output_dim, num_heads, device):
        super(GNNLayer, self).__init__()
        self.num_heads = num_heads
        self.special_sp_mm = SpecialSpmm()
        self.mlp_es = torch.nn.ModuleList()
        self.edge_wt = torch.nn.ModuleList()
        self.device = device

        for heads in range(num_heads):
            self.mlp_es.append(MLP(num_mlp_layers, input_dim, hidden_dim, output_dim))
            self.edge_wt.append(Attention(output_dim * 2 + 1))

    def forward(self, x, adj_block):
        idx, elem, shape = adj_block
        h = []
        num_r = x.shape[0]
        ones = torch.ones(size=(num_r, 1), device=self.device)
        for head in range(self.num_heads):
            features = self.mlp_es[head](x)
            x_cat = [features[idx[0]], features[idx[1]], elem.unsqueeze(1)]
            x_cat = torch.cat(x_cat, dim=1)
            elem_new1 = self.edge_wt[head](x_cat).squeeze(1)
            # elem_new = -F.relu(self.edge_wt[head](x_cat) / 20)
            elem_new = elem_new1 - torch.max(elem_new1)

            elem_new = torch.exp(elem_new)
            try:
                assert not torch.isnan(elem_new).any()
            except AssertionError:
                print(elem_new1)
            pooled = self.special_sp_mm(idx, elem_new, shape, features)
            row_sum = self.special_sp_mm(idx, elem_new, shape, ones) + 0.0000001
            pooled = pooled.div(row_sum)
            h.append(pooled)
        h = torch.cat(h, dim=1)
        # assert not torch.isnan(h).any()
        return h
