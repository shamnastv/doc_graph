import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp import MLP
from util import row_norm


class GNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps,
                 graph_pooling_type, neighbor_pooling_type, device, beta):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.w1 = nn.Parameter(torch.zeros(self.num_layers - 1))
        # self.w2 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.beta = beta

        # for layer in self.layers:
        #     if layer == 0:
        #         self.mlps_c.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
        #     else:
        #         self.mlps_c.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
        #     self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        ###List of MLPs
        self.mlp_es = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        # self.norm_g_embd = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlp_es.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                # self.norm_g_embd.append(nn.BatchNorm1d(input_dim))
            else:
                self.mlp_es.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                # self.norm_g_embd.append(nn.BatchNorm1d(hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.eps)
        nn.init.uniform_(self.w1)
        nn.init.uniform_(self.eps)

    def __preprocess_neighbors_maxpool(self, batch_graph):
        # create padded_neighbor_list in concatenated graph

        # compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                # add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                # padding, dummy data is assumed to be stored in -1
                pad.extend([-1] * (max_deg - len(pad)))

                # Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        # create block diagonal sparse matrix

        edge_mat_list = []
        edge_weight_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
            edge_weight_list.append(graph.edges_weights)
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.cat(edge_weight_list)
        # Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        # Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        # if not self.learn_eps:
        #     num_node = start_idx[-1]
        #     self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
        #     elem = torch.ones(num_node)
        #     Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
        #     Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device)

    def __preprocess_graphpool_n(self, batch_graph):
        # create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            elem.extend([1] * len(graph.g))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device).transpose(0, 1)

    def __preprocess_graphpool(self, batch_graph):
        # create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            # average pooling
            if self.graph_pooling_type == "average":
                # elem.extend([1. / len(graph.g)] * len(graph.g))
                elem.extend(graph.node_tags)
            else:
                # sum pooling
                elem.extend([1] * len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        # Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def next_layer_eps(self, h, layer, idx, Cl=None, ge=None, graph_pool_n=None, padded_neighbor_list=None, Adj_block=None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # Re-weights the center node representation when aggregating it with its neighbors
        # pooled = (1 + self.w1[layer]) * pooled + (1 + self.eps[layer]) * h
        if Cl is not None and False:
            # mul_fact = self.beta / H.shape[0]
            tmp = torch.mm(Cl[idx], Cl.transpose(0, 1))
            tmp = torch.spmm(tmp, ge)
            # tmp = row_norm(tmp)
            tmp = (self.beta + self.w1[layer]) * tmp
            # tmp = self.beta * tmp
            # if self.training:
            #     if bool(random.getrandbits(1)):
            #         tmp = 0 * tmp
            # else:
            #     tmp = .5 * tmp
            pooled = pooled + torch.spmm(graph_pool_n, tmp)
        pooled = pooled + (1 + self.eps[layer]) * h
        h = self.mlp_es[layer](pooled)
        h = self.batch_norms[layer](h)
        h = F.relu(h)
        # h = F.leaky_relu(h)
        # h = F.tanh(h)
        h = F.dropout(h, self.final_dropout, training=self.training)
        return h

    def next_layer(self, h, layer, idx, Cl=None, ge=None, graph_pool_n=None, padded_neighbor_list=None, Adj_block=None):
        # pooling neighboring nodes and center nodes  altogether

        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # representation of neighboring and center nodes
        if Cl is not None:
            tmp = torch.mm(Cl[idx], Cl.transpose(0, 1))
            tmp = torch.spmm(tmp, ge)
            pooled = pooled + torch.spmm(graph_pool_n, tmp)
        pooled_rep = self.mlp_es[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self, batch_graph, Cl, ge, idx):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)
        graph_pool_n = self.__preprocess_graphpool_n(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        # list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers - 1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, idx, Cl, ge[layer], graph_pool_n, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, idx, Cl, ge[layer], graph_pool_n, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, idx, Cl, ge[layer], graph_pool_n, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, idx, Cl, ge[layer], graph_pool_n, Adj_block=Adj_block)

            hidden_rep.append(h)

        score_over_layer = 0
        pooled_h_ls = []

        # perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            if Cl is not None:
                tmp = torch.mm(Cl[idx], Cl.transpose(0, 1))
                tmp = torch.spmm(tmp, ge[layer])
                # tmp = row_norm(tmp)
                tmp = (self.beta + self.w1[layer]) * tmp
                pooled_h = pooled_h + tmp
            # score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), .3,
            #                               training=self.training)
            # if layer == self.num_layers - 1:
            #     score_over_layer += self.linears_prediction[layer](pooled_h)
            score_over_layer += self.linears_prediction[layer](pooled_h)
            pooled_h_ls.append(pooled_h)

        return score_over_layer, pooled_h_ls
