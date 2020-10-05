import networkx as nx
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn.preprocessing import normalize

import build_graph
from cluster_mod import ClusterNN
from gnn_mod import GNN
from util import normalize_adj, row_norm

criterion = nn.CrossEntropyLoss()
frequency_as_feature = False
max_val_accuracy = 0
test_accuracy = 0
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
        # self.neighbors = []
        self.node_features = torch.FloatTensor(node_features)
        # self.node_features = row_norm(self.node_features)
        self.edge_mat = 0
        self.edges_weights = []

        # self.max_neighbor = 0


def create_gaph(args):
    ls_adj, feature_list, word_freq_list, y, y_hot, train_size = build_graph.build_graph(config_file=args.configfile)
    g_list = []
    for i, adj in enumerate(ls_adj):
        adj = normalize_adj(adj)
        g = nx.from_scipy_sparse_matrix(adj)
        lb = y[i]
        feat = feature_list[i]
        if frequency_as_feature:
            feat = np.concatenate((feat, word_freq_list[i].toarray()), axis=1)
            # feat = feat * word_freq_list[i].toarray()
        s = sum(word_freq_list[i])
        wf = [el/s for el in word_freq_list[i]]
        g_list.append(S2VGraph(g, lb, node_features=feat, node_tags=wf))

    for g in g_list:
        # g.neighbors = [[] for i in range(len(g.g))]
        # for i, j in g.g.edges():
        #     g.neighbors[i].append(j)
        #     g.neighbors[j].append(i)
        # degree_list = []
        # for i in range(len(g.g)):
        #     g.neighbors[i] = g.neighbors[i]
        #     degree_list.append(len(g.neighbors[i]))
        # g.max_neighbor = max(degree_list)
        edges = [list(pair) for pair in g.g.edges()]
        edges_w = [w['weight'] for i, j, w in g.g.edges(data=True)]
        edges.extend([[i, j] for j, i in edges])
        edges_w.extend([w for w in edges_w])
        if len(edges) == 0:
            edges = [[0, 0]]
            edges_w = [1]
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)
        g.edges_weights = torch.FloatTensor(edges_w)
    return g_list, len(set(y)), train_size


def my_loss(alpha, centroids, embeddings, cl, device):
    dm = len(cl[0])
    loss = 0
    for i, emb in enumerate(embeddings):
        tmp = torch.sum(torch.sub(centroids, emb) ** 2, dim=1, keepdim=True)
        # tmp = torch.sub(centroids, emb)
        # loss += torch.mm(cl[i].reshape(1, -1), torch.norm(tmp, dim=1, keepdim=True))
        loss += torch.mm(cl[i].reshape(1, -1), tmp)
    tmp = torch.mm(cl.transpose(0, 1), cl)
    loss += alpha * torch.norm(tmp / torch.norm(tmp) - torch.eye(dm).to(device) / (dm ** .5))
    return loss


def print_cluster(cl):
    freq = [0 for i in range(cl.shape[1])]
    indices = cl.max(1)[1]
    for i in indices:
        freq[i] += 1
    print(freq)


def train(args, model_e, model_c, device, graphs, optimizer, optimizer_c, epoch, train_size, ge, cl, initial=False):
    total_size = len(graphs)

    val_size = train_size // args.n_fold
    train_size = train_size - val_size
    test_size = total_size - train_size

    model_e.train()
    model_c.train()

    total_itr_c = args.iters_per_epoch
    cl_batch_size = args.batch_size_cl

    ge_new = []
    for layer in range(args.num_layers):
        if layer == 0:
            ge_new.append(torch.zeros(len(graphs), graphs[0].node_features.shape[1]).to(device))
        else:
            ge_new.append(torch.zeros(len(graphs), args.hidden_dim).to(device))

    if not initial:
        if epoch % total_itr_c == 1:
            for itr in range(total_itr_c):
                loss_c_accum = 0
                full_idx = np.random.permutation(total_size)
                num_itr = 0
                for i in range(0, total_size, cl_batch_size):
                    selected_idx = full_idx[i:i + cl_batch_size]
                    ge_tmp = [ge_t[selected_idx] for ge_t in ge]
                    cl_new = model_c(ge_tmp)
                    loss_c = 0
                    for layer in range(args.num_layers):
                        loss_c += my_loss(args.alpha, model_c.centroids[layer], ge_tmp[layer], cl_new, device)
                    if optimizer_c is not None:
                        optimizer_c.zero_grad()
                        loss_c.backward()
                        optimizer_c.step()
                    loss_c = loss_c.detach().cpu().numpy()
                    loss_c_accum += loss_c
                    cl_new = cl_new.detach()
                    num_itr += 1
                print('epoch : ', epoch, 'itr', itr, 'cluster loss : ', loss_c_accum/num_itr)
            model_c.eval()
            with torch.no_grad():
                cl = model_c(ge)
            print_cluster(cl)
            print('', flush=True)
        # else:
        #     with torch.no_grad():
        #         cl = model_c(ge)

    else:
        cl = None

    idx_train = np.random.permutation(train_size)
    loss_accum = 0
    for i in range(0, train_size, args.batch_size):
        selected_idx = idx_train[i:i + args.batch_size]
        batch_graph = [graphs[idx] for idx in selected_idx]
        if len(selected_idx) == 0:
            continue
        output, pooled_h = model_e(batch_graph, cl, ge, selected_idx)

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
        for layer in range(args.num_layers):
            ge_new[layer][selected_idx] = pooled_h[layer].detach()

    print('epoch : ', epoch, 'classification loss : ', loss_accum)

    with torch.no_grad():
        # model_e.eval()
        idx_test = np.arange(train_size, total_size)
        for i in range(0, test_size, args.batch_size):
            selected_idx = idx_test[i:i + args.batch_size]
            batch_graph = [graphs[idx] for idx in selected_idx]
            if len(selected_idx) == 0:
                continue
            output, pooled_h = model_e(batch_graph, cl, ge, selected_idx)

            for layer in range(args.num_layers):
                ge_new[layer][selected_idx] = pooled_h[layer]

    print(time.time() - start_time, 's Epoch : ', epoch, 'loss training: ', loss_accum)

    return loss_accum, ge_new, cl


# pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(args, model_e, graphs, cl, ge, minibatch_size, device):
    outputs = []
    ge_new = []
    for layer in range(args.num_layers):
        if layer == 0:
            ge_new.append(torch.zeros(len(graphs), graphs[0].node_features.shape[1]).to(device))
        else:
            ge_new.append(torch.zeros(len(graphs), args.hidden_dim).to(device))

    full_idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = full_idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        with torch.no_grad():
            output, pooled_h = model_e([graphs[j] for j in sampled_idx], cl, ge, sampled_idx)
        outputs.append(output)
        for layer in range(args.num_layers):
            ge_new[layer][sampled_idx] = pooled_h[layer]

    return torch.cat(outputs, 0), ge_new


def test(args, model_e, model_c, device, graphs, train_size, epoch, ge, cl):
    model_c.eval()
    model_e.eval()

    val_size = int(train_size / args.n_fold)
    train_size = train_size - val_size

    # cl = model_c(ge)

    output, ge_new = pass_data_iteratively(args, model_e, graphs, cl, ge, 100, device)

    output_train, output_val, output_test = output[:train_size], output[train_size:train_size + val_size], output[train_size + val_size:]
    train_graphs, val_graph, test_graphs = graphs[:train_size], graphs[train_size:train_size + val_size], graphs[train_size + val_size:]

    pred_train = output_train.max(1, keepdim=True)[1]
    labels_train = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred_train.eq(labels_train.view_as(pred_train)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    pred_val = output_val.max(1, keepdim=True)[1]
    labels_val = torch.LongTensor([graph.label for graph in val_graph]).to(device)
    correct = pred_val.eq(labels_val.view_as(pred_val)).sum().cpu().item()
    acc_val = correct / float(len(val_graph))

    pred_test = output_test.max(1, keepdim=True)[1]
    labels_test = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred_test.eq(labels_test.view_as(pred_test)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print(time.time() - start_time, 's epoch : ', epoch)
    print("accuracy train: %f val: %f test: %f" % (acc_train, acc_val, acc_test))
    global max_acc_epoch, max_val_accuracy, test_accuracy
    if acc_val > max_val_accuracy:
        max_val_accuracy = acc_val
        max_acc_epoch = epoch
        test_accuracy = acc_test

    print('max validation accuracy : ', max_val_accuracy, 'max acc epoch : ', max_acc_epoch, flush=True)
    print('epsilon : ', model_e.eps)
    print('w1 : ', model_e.w1)

    # if epoch == 800:
    #     for i in range(len(test_graphs)):
    #         print('label : ', labels_test[i].cpu().item(), ' pred : ', pred_test[i].cpu().item())

    return acc_train, acc_test, ge_new


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--batch_size_cl', type=int, default=500,
                        help='input batch size for clustering (default: 32)')
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
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
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
    parser.add_argument('--beta', type=float, default=1,
                        help='beta')
    parser.add_argument('--n_fold', type=float, default=5,
                        help='n_fold')

    args = parser.parse_args()

    print(args)

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    print('device : ', device, flush=True)

    graphs, num_classes, train_size = create_gaph(args)
    ge = [None for i in range(args.num_layers)]

    model_c = ClusterNN(num_classes, graphs[0].node_features.shape[1], args.hidden_dim, args.num_layers,
                        args.num_mlp_layers_c).to(device)
    model_e = GNN(args.num_layers, args.num_mlp_layers, graphs[0].node_features.shape[1], args.hidden_dim, num_classes,
                  args.final_dropout,
                  args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device, args.beta).to(device)

    optimizer = optim.Adam(model_e.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(model_c.parameters(), lr=args.lr_c)
    # optimizer = optim.SGD(model_e.parameters(), lr=args.lr, momentum=0.9)
    # optimizer_c = optim.SGD(model_c.parameters(), lr=args.lr_c, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    cl = None

    print(time.time() - start_time, 's Training starts', flush=True)
    for epoch in range(10):
        avg_loss, ge_new, cl = train(args, model_e, model_c, device, graphs, optimizer, optimizer_c, epoch,
                                 train_size, ge, cl, initial=True)
    print('Embedding Initialized', flush=True)
    # acc_train, acc_test, ge_new = test(args, model_e, model_c, device, graphs, train_size, 10, ge)

    # for i in range(len(ge)):
    #     ge[i] = row_norm(ge_new[i])
    ge = ge_new

    for epoch in range(1, args.epochs + 1):
        avg_loss, ge_new, cl = train(args, model_e, model_c, device, graphs, optimizer,
                                 optimizer_c, epoch, train_size, ge, cl)
        acc_train, acc_test, ge_new = test(args, model_e, model_c, device, graphs, train_size, epoch, ge, cl)
        scheduler.step()

        if epoch % args.iters_per_epoch == 0 or True:
            # for i in range(len(ge)):
            #     ge[i] = row_norm(ge_new[i])
            ge = ge_new

            # model_c = ClusterNN(num_classes, graphs[0].node_features.shape[1], args.hidden_dim, args.num_layers,
            #                     args.num_mlp_layers_c).to(device)
            # model_e = GNN(args.num_layers, args.num_mlp_layers, graphs[0].node_features.shape[1], args.hidden_dim,
            #               num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
            #               args.neighbor_pooling_type, device, args.beta).to(device)
            #
            # optimizer = optim.Adam(model_e.parameters(), lr=args.lr)
            # optimizer_c = optim.Adam(model_c.parameters(), lr=args.lr_c)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            print(time.time() - start_time, 'embeddings updated.', flush=True)

        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        print("")

    print(time.time() - start_time, 's Completed')
    print('total size : ', len(graphs))
    print('max validation accuracy : ', max_val_accuracy)
    print('max acc epoch : ', max_acc_epoch)
    print('test accuracy : ', test_accuracy)


if __name__ == '__main__':
    main()
