import networkx as nx
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time

import build_graph
from gnn import GNN
from util import normalize_adj

criterion = nn.CrossEntropyLoss()
d = torch.device('cpu')
frequency_as_feature = False
max_val_accuracy = 0
test_accuracy = 0
max_acc_epoch = 0
start_time = time.time()


class S2VGraph(object):
    def __init__(self, g, label, word_freq=None, node_features=None, positions=None):
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
        self.word_freq = word_freq
        self.node_features = node_features
        self.positions = positions
        # self.node_features = row_norm(self.node_features)
        self.edge_mat = 0
        self.edges_weights = []


def print_distr(y, train_size):
    print('Class distributions')
    freq = [0 for i in range(len(set(y)))]
    for i in y[:train_size]:
        freq[i] += 1
    print('on train : ', freq)
    m = 1
    for i in freq:
        if i < m:
            m = i
    weights = [m / i if i != 0 else 1 for i in freq]
    global criterion
    # criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(d))
    freq = [0 for i in range(len(set(y)))]
    for i in y[train_size:]:
        freq[i] += 1
    print('on test : ', freq)


def create_gaph(args):
    ls_adj, feature_list, word_freq_list, y, y_hot, train_size, word_vectors, positions_list = build_graph.build_graph(
        config=args.configfile)
    word_vectors = torch.from_numpy(word_vectors).float()

    print_distr(y, train_size)
    g_list = []
    max_words = 0
    for i, adj in enumerate(ls_adj):
        adj = normalize_adj(adj)
        g = nx.from_scipy_sparse_matrix(adj)
        lb = y[i]
        feat = feature_list[i]
        # if frequency_as_feature:
        #     feat = np.concatenate((feat, word_freq_list[i].toarray()), axis=1)
        #     # feat = feat * word_freq_list[i].toarray()
        if i == 10:
            print(word_freq_list[i])
        s = sum(word_freq_list[i])
        # s = 1
        wf = [el / s for el in word_freq_list[i]]
        g_list.append(S2VGraph(g, lb, node_features=feat, word_freq=wf, positions=positions_list[i]))
        for ar in positions_list[i]:
            max_words = max(max_words, max(ar))

    max_words += 1

    zero_edges = 0
    for g in g_list:
        # edges = [list(pair) for pair in g.g.edges()]
        # edges_w = [w['weight'] for i, j, w in g.g.edges(data=True)]
        # edges.extend([[i, j] for j, i in edges])
        # edges_w.extend([w for w in edges_w])
        edges = []
        edges_w = []
        for i, j, wt in g.g.edges(data=True):
            w = wt['weight']
            edges.append([i, j])
            edges_w.append(w)
            if i != j:
                edges.append([j, i])
                edges_w.append(w)

        if len(edges) == 0:
            # print('zero edge : ', len(g.g))
            zero_edges += 1
            edges = [[0, 0]]
            edges_w = [0]
        g.edge_mat = torch.tensor(edges).long().transpose(0, 1)
        g.edges_weights = torch.tensor(edges_w).float()
    print('total zero edge graphs : ', zero_edges)
    return g_list, len(set(y)), train_size, word_vectors, max_words


def train(args, model_e, device, graphs, optimizer, epoch, train_size, word_vectors):
    model_e.train()
    total_size = len(graphs)

    val_size = train_size // args.k_fold
    train_size = train_size - val_size
    test_size = total_size - train_size

    idx_train = np.random.permutation(train_size)
    loss_accum = 0
    for i in range(0, train_size, args.batch_size):
        selected_idx = idx_train[i:i + args.batch_size]
        batch_graph = [graphs[idx] for idx in selected_idx]
        # if len(selected_idx) == 0:
        #     continue
        output = model_e(batch_graph, word_vectors)

        labels = torch.tensor([graph.label for graph in batch_graph]).long().to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

    print('Epoch : ', epoch, 'loss training: ', loss_accum, 'Time : ', int(time.time() - start_time))
    print(model_e.pos)
    return loss_accum


# pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(args, model_e, graphs, minibatch_size, device, word_vectors):
    outputs = []
    full_idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = full_idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        with torch.no_grad():
            output = model_e([graphs[j] for j in sampled_idx], word_vectors)
        outputs.append(output)

    return torch.cat(outputs, 0)


def test(args, model_e, device, graphs, train_size, epoch, word_vectors):
    model_e.eval()

    val_size = train_size // args.k_fold
    train_size = train_size - val_size

    output = pass_data_iteratively(args, model_e, graphs, 100, device, word_vectors)

    output_train, train_graphs = output[:train_size], graphs[:train_size]
    output_val, val_graph = output[train_size:train_size + val_size], graphs[train_size:train_size + val_size]
    output_test, test_graphs = output[train_size + val_size:], graphs[train_size + val_size:]

    pred_train = output_train.max(1, keepdim=True)[1]
    labels_train = torch.tensor([graph.label for graph in train_graphs]).long().to(device)
    correct = pred_train.eq(labels_train.view_as(pred_train)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    pred_val = output_val.max(1, keepdim=True)[1]
    labels_val = torch.tensor([graph.label for graph in val_graph]).long().to(device)
    correct = pred_val.eq(labels_val.view_as(pred_val)).sum().cpu().item()
    acc_val = correct / float(len(val_graph))

    pred_test = output_test.max(1, keepdim=True)[1]
    labels_test = torch.tensor([graph.label for graph in test_graphs]).long().to(device)
    correct = pred_test.eq(labels_test.view_as(pred_test)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    if args.debug:
        print(time.time() - start_time, 's epoch : ', epoch)
    print("accuracy train: %f val: %f test: %f" % (acc_train, acc_val, acc_test), flush=True)
    global max_acc_epoch, max_val_accuracy, test_accuracy
    if acc_val > max_val_accuracy:
        max_val_accuracy = acc_val
        max_acc_epoch = epoch
        test_accuracy = acc_test

    print('max validation accuracy : ', max_val_accuracy, 'max acc epoch : ', max_acc_epoch, flush=True)
    if args.debug:
        print('epsilon : ', model_e.eps.detach().cpu().numpy(), 'w1 : ', model_e.w1.detach().cpu().numpy(), flush=True)

        # if epoch == 800:
        #     for i in range(len(test_graphs)):
        #         print('label : ', labels_test[i].cpu().item(), ' pred : ', pred_test[i].cpu().item())

    return acc_train, acc_test


def main():
    print('date and time : ', time.ctime())
    global max_acc_epoch, max_val_accuracy, test_accuracy
    acc_test = 0
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="average", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training '
                             'accuracy though.')
    parser.add_argument('--configfile', type=str, default="param",
                        help='configuration file')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    parser.add_argument('--k_fold', type=int, default=5,
                        help='k_fold')
    parser.add_argument('--early_stop', type=int, default=30,
                        help='early_stop')
    parser.add_argument('--debug', action="store_true",
                        help='run in debug mode')
    parser.add_argument('--num_heads', type=int, default=3,
                        help='number of hidden units (default: 64)')

    args = parser.parse_args()

    print(args)

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    print('device : ', device, flush=True)

    global d
    d = device
    all_graphs, num_classes, train_size, word_vectors, max_words = create_gaph(args)

    acc_detais = []
    k_start = 0
    if args.k_fold == 0:
        k_start = 9
        args.k_fold = 10
    val_size = train_size // args.k_fold
    for k in range(k_start, args.k_fold):
        start = k * val_size
        end = start + val_size
        graphs = all_graphs[:start] + all_graphs[end: train_size] + all_graphs[start:end] + all_graphs[train_size:]

        # ge = [None for i in range(args.num_layers)]

        # model_c = ClusterNN(num_classes, graphs[0].node_features.shape[1], args.hidden_dim, args.num_layers,
        #                     args.num_mlp_layers_c).to(device)
        model_e = GNN(args.num_layers, args.num_mlp_layers, word_vectors.shape[1], args.hidden_dim,
                      num_classes,
                      args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type,
                      device, max_words, args.num_heads, word_vectors).to(device)

        optimizer = optim.Adam(model_e.parameters(), lr=args.lr)
        # optimizer_c = optim.Adam(model_c.parameters(), lr=args.lr_c)
        # optimizer = optim.SGD(model_e.parameters(), lr=args.lr, momentum=0.9)
        # optimizer_c = optim.SGD(model_c.parameters(), lr=args.lr_c, momentum=0.9)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        # cl = None

        print(time.time() - start_time, 's Training starts', flush=True)
        for epoch in range(1, args.epochs + 1):
            avg_loss = train(args, model_e, device, graphs, optimizer, epoch, train_size, word_vectors)
            acc_train, acc_test = test(args, model_e, device, graphs, train_size, epoch, word_vectors)
            print("")

            if epoch > max_acc_epoch + args.early_stop \
                    and epoch > args.early_stop:
                break

        print('=' * 200)
        print('K : ', k, 'Time : ', abs(time.time() - start_time))
        print('max validation accuracy : ', max_val_accuracy * 100)
        print('max acc epoch : ', max_acc_epoch)
        print('test accuracy : ', test_accuracy * 100)
        print('latest_test_accuracy : ', acc_test * 100)
        print('=' * 200 + '\n')
        acc_detais.append((max_val_accuracy, test_accuracy, max_acc_epoch, acc_test))
        max_val_accuracy = 0
        test_accuracy = 0
        max_acc_epoch = 0

    print(args)
    print('total size : ', len(all_graphs), '\n')
    print('=' * 71 + 'Summary' + '=' * 71)
    for k in range(len(acc_detais)):
        print('k : ', k,
              '\tval_accuracy : ', acc_detais[k][0] * 100,
              '\ttest_accuracy : ', acc_detais[k][1] * 100,
              '\tmax_acc epoch : ', acc_detais[k][2],
              '\tlatest_test_accuracy : ', acc_detais[k][3] * 100)


if __name__ == '__main__':
    main()
