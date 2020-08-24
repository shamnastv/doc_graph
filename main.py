import networkx as nx
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import build_graph
from cluster import ClusterNN
from gnn import GNN

criterion = nn.CrossEntropyLoss()
frequency_as_feature = True
max_test_accuracy = 0
max_acc_epoch = 0
start_time = time.time()


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = torch.FloatTensor(node_features)
        self.edge_mat = 0

        self.max_neighbor = 0


def create_gaph(args):
    ls_adj, feature_list, word_freq_list, y, y_hot, train_size = build_graph.build_graph(config_file=args.configfile)
    g_list = []
    for i, adj in enumerate(ls_adj):
        g = nx.from_scipy_sparse_matrix(adj)
        lb = y[i]
        feat = feature_list[i]
        if frequency_as_feature:
            # feat = np.concatenate((feat, word_freq_list[i].toarray()), axis=1)
            feat = feat * word_freq_list[i].toarray()
        g_list.append(S2VGraph(g, lb, node_features=feat))

    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    return g_list, len(set(y)), train_size


def my_loss(alpha, centroids, embeddings, cl, device):
    dm = len(cl[0])
    loss = 0
    for i, emb in enumerate(embeddings):
        tmp = torch.sub(centroids, emb)
        loss += torch.mm(cl[i].reshape(1, -1), torch.norm(tmp, dim=1, keepdim=True))
    tmp = torch.mm(cl.transpose(0, 1), cl)
    loss += alpha * torch.norm(tmp / torch.norm(tmp) - torch.eye(dm).to(device) / dm ** .5)
    return loss


def train(args, model_e, model_c, device, graphs, optimizer, optimizer_c, epoch, train_size, ge):
    model_e.train()
    model_c.train()

    total_iter = args.iters_per_epoch + 2 * epoch
    total_iter_c = total_iter

    if epoch >= 21:
        total_iter = 2

    if epoch >= 22 and (not epoch % 30 == 0):
        total_iter_c = 1
        optimizer_c = None

    node_features = [0 for i in range(len(graphs))]
    ge_new = torch.zeros(len(graphs), graphs[0].node_features.shape[1]).to(device)

    for itr in range(total_iter_c):
        cl = model_c(ge)
        loss_c = my_loss(args.alpha, model_c.centroids, ge, cl, device)
        if optimizer_c is not None:
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()
        cl = cl.detach()
        print('epoch : ', epoch, 'itr : ', itr, 'cluster loss : ', loss_c.detach().cpu().numpy())

    idx_train = np.random.permutation(train_size)
    for itr in range(total_iter):
        loss_accum = 0
        for i in range(0, train_size, args.batch_size):
            selected_idx = idx_train[i:i + args.batch_size]
            batch_graph = [graphs[idx] for idx in selected_idx]
            if len(selected_idx) == 0:
                continue
            output, pooled_h, h = model_e(batch_graph, cl, ge, selected_idx)

            labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

            # compute loss
            loss = criterion(output, labels)

            # backprop
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = loss.detach().cpu().numpy()
            loss_accum += loss

            ge_new[selected_idx] = pooled_h.detach()
            h = h.detach()
            start_idx = 0
            if itr == total_iter - 1:
                for j in selected_idx:
                    length = len(graphs[j].g)
                    node_features[j] = h[start_idx:start_idx + length]
                    start_idx += length

        print('epoch : ', epoch, 'itr : ', itr, 'classification loss : ', loss_accum, 'W : ', model_e.ws)
    model_e.eval()
    total_size = len(graphs)
    test_size = total_size - train_size
    idx_test = np.arange(train_size, total_size)
    for i in range(0, test_size, args.batch_size):
        selected_idx = idx_test[i:i + args.batch_size]
        batch_graph = [graphs[idx] for idx in selected_idx]
        if len(selected_idx) == 0:
            continue
        output, pooled_h, h = model_e(batch_graph, cl, ge, selected_idx)

        output = output.detach()
        ge_new[selected_idx] = pooled_h.detach()
        h = h.detach()
        start_idx = 0
        for j in selected_idx:
            length = len(graphs[j].g)
            node_features[j] = h[start_idx:start_idx + length]
            start_idx += length

    print(time.time() - start_time, 's Epoch : ', epoch, 'loss training: ', loss_accum)

    return loss_accum, ge_new, node_features


# pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model_e, graphs, cl, ge, minibatch_size=64):
    outputs = []
    full_idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = full_idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output, pooled_h, h = model_e([graphs[j] for j in sampled_idx], cl, ge, sampled_idx)
        outputs.append(output.detach())
    return torch.cat(outputs, 0)


def test(args, model_e, model_c, device, graphs, train_size, epoch, ge):
    model_c.eval()
    model_e.eval()

    cl = model_c(ge)

    output = pass_data_iteratively(model_e, graphs, cl, ge)

    output_train, output_test = output[:train_size], output[train_size:]
    train_graphs, test_graphs = graphs[:train_size], graphs[train_size:]

    pred_train = output_train.max(1, keepdim=True)[1]
    labels_train = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred_train.eq(labels_train.view_as(pred_train)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    pred_test = output_test.max(1, keepdim=True)[1]
    labels_test = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred_test.eq(labels_test.view_as(pred_test)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print(time.time() - start_time, 's epoch : ', epoch)
    print("accuracy train: %f test: %f" % (acc_train, acc_test))
    global max_acc_epoch, max_test_accuracy
    if acc_test > max_test_accuracy:
        max_test_accuracy = acc_test
        max_acc_epoch = epoch

    if epoch == 800:
        for i in range(len(test_graphs)):
            print('label : ', labels_test[i], ' pred : ', pred_test[i])

    return acc_train, acc_test


def initialize_graph_embedding(graphs, device):
    embeddings = torch.zeros(len(graphs), graphs[0].node_features.shape[1]).to(device)
    for i, g in enumerate(graphs):
        embeddings[i] = g.node_features.mean(dim=0).to(device)

    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_c', type=float, default=0.01,
                        help='learning rate for clustering (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--num_mlp_layers_c', type=int, default=2,
                        help='number of layers for MLP clustering EXCLUDING the input one (default: 2). 1 means '
                             'linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training '
                             'accuracy though.')
    parser.add_argument('--configfile', type=str, default="param.yaml",
                        help='configuration file')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    parser.add_argument('--alpha', type=float, default=1,
                        help='alpha')

    args = parser.parse_args()

    print(args)

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes, train_size = create_gaph(args)
    ge = initialize_graph_embedding(graphs, device)

    model_c = ClusterNN(num_classes, ge.shape[1], args.num_mlp_layers_c).to(device)
    model_e = GNN(args.num_mlp_layers, graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout,
                args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model_e.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(model_c.parameters(), lr=args.lr_c)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    print(time.time() - start_time, 's Training starts')
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss, ge_new, node_features = train(args, model_e, model_c, device, graphs, optimizer, optimizer_c, epoch, train_size, ge)
        acc_train, acc_test = test(args, model_e, model_c, device, graphs, train_size, epoch, ge)

        # update_graph = True
        update_graph = ((epoch % 10 == 0) and epoch <= 20)
        if update_graph:
            for j in range(len(graphs)):
                graphs[j].node_features = node_features[j]
            ge = ge_new
            print('graph updated in epoch : ', epoch)

        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        print("")

        # print(model.eps)
    print(time.time() - start_time, 's Completed')
    print('total size : ', len(graphs))
    print('max test accuracy : ', max_test_accuracy)
    print('max acc epoch : ', max_acc_epoch)


if __name__ == '__main__':
    main()
