import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp import MLP


class ClusterNN(nn.Module):
    def __init__(self, num_class, input_dim, hidden_dim, num_layers, num_mlp_layers):
        super(ClusterNN, self).__init__()
        self.num_layers = num_layers
        self.score_of_layers = nn.Parameter(torch.zeros(self.num_layers))
        self.centroids = nn.ParameterList()
        for layer in range(num_layers):
            if layer == 0:
                self.centroids.append(nn.Parameter(torch.zeros(num_class, input_dim)))
            else:
                self.centroids.append(nn.Parameter(torch.zeros(num_class, hidden_dim)))

        self.mlp_cs = torch.nn.ModuleList()
        self.batch_norms_c = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.mlp_cs.append(MLP(num_mlp_layers, input_dim, hidden_dim, num_class))
            else:
                self.mlp_cs.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, num_class))
            self.batch_norms_c.append(nn.BatchNorm1d(num_class))

        # self.linears_prediction = torch.nn.ModuleList()
        # for layer in range(num_layers - 1):
        #     self.linears_prediction.append(nn.Linear(num_class, num_class))

        # self.batch_norm_c = nn.BatchNorm1d(num_class)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for centroid in self.centroids:
            nn.init.uniform_(centroid)
        nn.init.uniform_(self.score_of_layers)

    def forward(self, ge):
        cg = 0
        for layer in range(0, self.num_layers):
            # cg += self.batch_norms_c[layer](self.mlp_cs[layer](ge[layer]))
            cg += self.mlp_cs[layer](ge[layer])

        # cg = self.batch_norm_c(cg)
        cg = F.softmax(cg, dim=-1)
        return cg
