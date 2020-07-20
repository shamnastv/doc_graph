import random
import numpy as np
import scipy.sparse as sp
from math import log

dataset = '20ng'


doc_name_list = []
doc_train_list = []
doc_test_list = []

# Read meta data
f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()

# Read data from clean file
doc_content_list = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()

# Find train ids
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

# Find test ids
test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

ids = train_ids + test_ids
train_size = len(train_ids)
total_size = len(ids)

# Organize train and test data
shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])

# Creating labels
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

y = []
for i in range(total_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for lb in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)

word_freq_list = []
ls_adj = []
window_size = 20

index = 0
for doc_words in shuffle_doc_words_list:
    windows = []
    words = doc_words.split()

    # Find Word list and frequency
    word_freq = {}
    word_set = set()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

    vocab = list(word_set)
    vocab_size = len(vocab)

    # Create map of word to id
    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    # Create node attribute
    node_size = vocab_size
    row = []
    col = []
    val = []

    for key in word_freq:
        row.append(word_id_map[key])
        col.append(0)
        val.append(word_freq[key])
    feat = sp.csr_matrix(
        (val, (row, col)), shape=(node_size, 1))
    word_freq_list.append(feat)

    # Create windows
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

    # Find Word window frequency
    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    # Count word pair
    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    # Create adjacency matrix
    row = []
    col = []
    weight = []
    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(i)
        col.append(j)
        weight.append(count)

    node_size = vocab_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))

    ls_adj.append(adj)
    # print(adj.shape)
    if index in [0, 3]:
        print('Index : ', index)
        print(shuffle_doc_name_list[index])
        print(doc_words)
        print(word_id_map)
        print(feat)
        print(adj)
    index += 1

print('Shape of feature index 10 ', word_freq[10].shape)
print('Shape of adj index 10 ', ls_adj[10].shape)

print('Size of features ', len(word_freq))
print('Size of adjacency ', len(ls_adj))
print('Total Size ', total_size)