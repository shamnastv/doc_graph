import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from attention import Attention
from layer import GNNLayer
from torch_sparse import spmm


class GNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps,
                 graph_pooling_type, neighbor_pooling_type, device, max_words, num_heads, word_embeddings):
        """
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes.
                        If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        """

        super(GNN, self).__init__()

        self.input_dim = input_dim
        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        # self.eps = nn.Parameter(torch.zeros(self.num_layers - 1), requires_grad=True)
        # self.w1 = nn.Parameter(torch.zeros(self.num_layers), requires_grad=True)
        self.pos = nn.Parameter(torch.zeros(self.num_layers), requires_grad=True)
        self.do_once = True
        self.num_heads = num_heads

        self.dropout = nn.Dropout(final_dropout)

        self.real_hidden_dim = num_heads * hidden_dim

        self.positional_embeddings = np.zeros((max_words, self.input_dim))

        for position in range(max_words):
            for i in range(0, self.input_dim, 2):
                self.positional_embeddings[position, i] = (
                    np.sin(position / (10000 ** ((2 * i) / self.input_dim)))
                )
                self.positional_embeddings[position, i + 1] = (
                    np.cos(position / (10000 ** ((2 * (i + 1)) / self.input_dim)))
                )

        # List of MLPs
        self.mlp_es = torch.nn.ModuleList()
        self.gnn_layers = torch.nn.ModuleList()

        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        # self.norm_g_embd = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                # self.mlp_es.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                self.gnn_layers.append(GNNLayer(num_mlp_layers, input_dim, hidden_dim, hidden_dim, num_heads, device))
            else:
                # self.mlp_es.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                self.gnn_layers.append(GNNLayer(num_mlp_layers, self.real_hidden_dim,
                                                hidden_dim, hidden_dim, num_heads, device))

            self.batch_norms.append(nn.BatchNorm1d(self.real_hidden_dim))

        # Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        self.graph_pool_layer = torch.nn.ModuleList()
        # self.edge_wt = torch.nn.ModuleList()

        # self.cluster_cat = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
                # self.graph_pool_layer.append(Attention(input_dim + 1))
                self.graph_pool_layer.append(Attention(input_dim + 2))
                # self.edge_wt.append(Attention(input_dim * 2 + 1))
                # self.cluster_cat.append(nn.Linear(input_dim * 2, 1))
            else:
                self.linears_prediction.append(nn.Linear(self.real_hidden_dim, output_dim))
                # self.graph_pool_layer.append(Attention(self.real_hidden_dim + 1))
                self.graph_pool_layer.append(Attention(self.real_hidden_dim + 2))
                # self.edge_wt.append(Attention(input_dim * 2 + 1))
                # self.cluster_cat.append(nn.Linear(hidden_dim * 2, 1))

        self.word_embeddings = nn.Embedding(word_embeddings.shape[0], word_embeddings.shape[1])
        self.word_embeddings.weight.data.copy_(word_embeddings)
        self.word_embeddings.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # nn.init.uniform_(self.eps)
        # nn.init.uniform_(self.w1)
        nn.init.uniform_(self.pos)

    def get_pos_enc(self, positions):
        pos_enc = np.zeros((len(positions), self.input_dim))
        for i in range(len(positions)):
            for j in positions[i]:
                pos_enc[i] += self.positional_embeddings[j]
            pos_enc[i] = pos_enc[i] / len(positions[i])
        return torch.from_numpy(pos_enc).float()

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        # create block diagonal sparse matrix

        edge_mat_list = []
        edge_weight_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
            edge_weight_list.append(graph.edges_weights)

        adj_block_idx = torch.cat(edge_mat_list, 1).to(self.device)
        adj_block_elem = torch.cat(edge_weight_list).to(self.device)

        # adj_block_elem = torch.ones(adj_block_idx.shape[1])

        # Add self-loops in the adjacency matrix if learn_eps is False,
        # i.e., aggregate center nodes and neighbor nodes altogether.

        # if not self.learn_eps:
        #     num_node = start_idx[-1]
        #     self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
        #     elem = torch.ones(num_node)
        #     adj_block_idx = torch.cat([adj_block_idx, self_loop_edge], 1)
        #     adj_block_elem = torch.cat([adj_block_elem, elem], 0)

        # adj_block = torch.sparse.FloatTensor(adj_block_idx, adj_block_elem,
        # torch.Size([start_idx[-1], start_idx[-1]]))
        adj_block = adj_block_idx, adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]])
        return adj_block

    def get_adj(self, layer, adj_block, features):
        idx, elem, shape = adj_block
        features = [features[idx[0]], features[idx[1]], elem.unsqueeze(1)]
        features = torch.cat(features, dim=1)
        elem = torch.exp(-F.leaky_relu(self.edge_wt[layer](features) / 20).squeeze(1))
        assert not torch.isnan(elem).any()
        return idx, elem, shape

    def __preprocess_graphpool(self, batch_graph):
        # create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        elem1 = []
        elem2 = []
        for i, graph in enumerate(batch_graph):
            # average pooling
            # if self.graph_pooling_type == "average":
            #     # elem.extend([1. / len(graph.g)] * len(graph.g))
            #     elem.extend(graph.word_freq)
            # else:
            #     # sum pooling
            #     elem.extend([1] * len(graph.g))
            # elem.extend(graph.word_freq)
            elem1.extend(graph.word_freq1)
            elem2.extend(graph.word_freq2)
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        # elem = torch.tensor(elem).float().to(self.device)

        elem1 = torch.tensor(elem1).float().to(self.device)
        elem2 = torch.tensor(elem2).float().to(self.device)

        idx = torch.tensor(idx).long().transpose(0, 1).to(self.device)
        # graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        # graph_pool = (idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        graph_pool = (idx, elem1, elem2, torch.Size([len(batch_graph), start_idx[-1]]))

        # return graph_pool.to(self.device), elem.reshape(-1, 1).to(self.device)
        return graph_pool

    def next_layer_eps(self, h, layer, adj_block=None):

        # idx, elem, shape = self.get_adj(layer, Adj_block, h)
        # pooled = self.special_spmm(idx, elem, shape, h)
        # row_sum = self.special_spmm(idx, elem, shape, torch.ones(size=(h.shape[0], 1), device=self.device))
        # pooled = pooled.div(row_sum)
        h = self.gnn_layers[layer](h, adj_block)
        # pooled = pooled + (1 + self.eps[layer]) * h
        # h = self.mlp_es[layer](pooled)
        # h = self.mlp_es[layer](h, Adj_block)
        # h = F.relu(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.batch_norms[layer](h)
        return h

    def forward(self, batch_graph, word_vectors):
        # graph_pool, node_weights = self.__preprocess_graphpool(batch_graph)
        # idx_gp, elem_gp, shape_gp = self.__preprocess_graphpool(batch_graph)
        idx_gp, elem_gp1, elem_gp2, shape_gp = self.__preprocess_graphpool(batch_graph)

        # list of hidden representation at each layer (including input)
        node_ids = []
        positional_encoding = []
        for graph in batch_graph:
            node_ids.extend(graph.node_features)
            positional_encoding.append(self.get_pos_enc(graph.positions))

        positional_encoding = torch.cat(positional_encoding, dim=0).to(self.device)

        # h = word_vectors[node_ids].to(self.device)
        h = self.word_embeddings(torch.tensor(node_ids, device=self.device, dtype=torch.long))

        h = h + self.pos[0] * positional_encoding
        hidden_rep = [self.dropout(h)]

        adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        for layer in range(self.num_layers - 1):
            h = self.next_layer_eps(h, layer, adj_block=adj_block)
            hidden_rep.append(h)

        # graph_pool = graph_pool.to_dense()
        score_over_layer = 0

        pooled_h_ls = []
        # perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            # if self.do_once:
            #     print(graph_pool)
            #
            # tmp = torch.cat((h, elem_gp.unsqueeze(1)), dim=1)
            tmp = torch.cat((h, elem_gp1.unsqueeze(1), elem_gp2.unsqueeze(1)), dim=1)
            elem_gp = self.graph_pool_layer[layer](tmp).squeeze(1)

            with torch.no_grad():
                maximum = torch.max(elem_gp)

            elem_gp = elem_gp - maximum
            elem_gp = torch.exp(elem_gp)
            assert not torch.isnan(elem_gp).any()

            # elem_gp = torch.sigmoid(self.graph_pool_layer[layer](tmp).squeeze(1)) * elem_gp

            row_sum = spmm(idx_gp, elem_gp, shape_gp[0], shape_gp[1],
                           torch.ones(size=(h.shape[0], 1), device=self.device))
            # row_sum = spmm(idx_gp, elem_gp, shape_gp[0], shape_gp[1],
            #                torch.ones(size=(h.shape[0], 1), device=self.device))

            # pooled_h = self.special_spmm(idx_gp, elem_gp, shape_gp, h)
            pooled_h = spmm(idx_gp, elem_gp, shape_gp[0], shape_gp[1], h)
            assert not torch.isnan(pooled_h).any()

            pooled_h = pooled_h.div(row_sum + 1e-10)
            assert not torch.isnan(pooled_h).any()

            pooled_h_ls.append(pooled_h)
            # if self.do_once:
            #     print(graph_pool)
            #     self.do_once = False
            score_over_layer += self.linears_prediction[layer](pooled_h)

        return score_over_layer, pooled_h_ls
