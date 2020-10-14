import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp import MLP


class GNN(nn.Module):
    def __init__(self, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps,
                 graph_pooling_type, neighbor_pooling_type, device):
        '''
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
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(1))

        self.ws = nn.Parameter(torch.zeros(2))
        self.mlp_e = MLP(num_mlp_layers, input_dim, hidden_dim, input_dim)
        self.batch_norms_e = nn.BatchNorm1d(input_dim)

        self.linear_prediction = nn.Linear(input_dim, output_dim)

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
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        # Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device)

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
                elem.extend([1. / len(graph.g)] * len(graph.g))

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

    def next_layer_eps(self, h, idx, Cl=None, H=None, graph_pool=None, padded_neighbor_list=None, Adj_block=None):
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
        if Cl is not None:
            pooled = (1 + self.ws[0]) * pooled + (1 + self.eps) * h
            tmp = torch.mm(Cl[idx], Cl.transpose(0, 1))
            tmp = torch.spmm(tmp, H)
            pooled = pooled + (1 + self.ws[1]) * torch.spmm(graph_pool.transpose(0, 1), tmp)
        pooled_rep = self.mlp_e(pooled)
        h = self.batch_norms_e(pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, idx, Cl=None, H=None, graph_pool=None, padded_neighbor_list=None, Adj_block=None):
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
            pooled = (1 + self.ws[0]) * pooled + (1 + self.eps) * h
            tmp = torch.mm(Cl[idx], Cl.transpose(0, 1))
            tmp = torch.spmm(tmp, H)
            pooled = pooled + (1 + self.ws[1]) * (torch.spmm(graph_pool.transpose(0, 1), tmp))
        pooled_rep = self.mlp_e(pooled)
        h = self.batch_norms_e(pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self, batch_graph, Cl, H, idx):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        # list of hidden representation at each layer (including input)
        h = X_concat

        if self.neighbor_pooling_type == "max" and self.learn_eps:
            h = self.next_layer_eps(h, idx, Cl, H, graph_pool, padded_neighbor_list=padded_neighbor_list)
        elif not self.neighbor_pooling_type == "max" and self.learn_eps:
            h = self.next_layer_eps(h, idx, Cl, H, graph_pool, Adj_block=Adj_block)
        elif self.neighbor_pooling_type == "max" and not self.learn_eps:
            h = self.next_layer(h, idx, Cl, H, graph_pool, padded_neighbor_list=padded_neighbor_list)
        elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
            h = self.next_layer(h, Cl, idx, H, graph_pool, Adj_block=Adj_block)

        pooled_h = torch.spmm(graph_pool, h)
        score_over_layer = F.dropout(self.linear_prediction(pooled_h), self.final_dropout,
                                     training=self.training)

        return score_over_layer, pooled_h, h
