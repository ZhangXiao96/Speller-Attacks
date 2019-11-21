import numpy as np
import math
from sklearn.metrics import confusion_matrix


def bca(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i]/np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label/numb


def ITR(p, n, t):
    """
    Calculate the Information Translate Rate (ITR).
    ITR=\frac{60}{t}[log_2n+plog_2p+(1-p)log_2\frac{1-p}{Qpn-1}]
    :param p: Accuracy.
    :param n: Number of targets (classes).
    :param t: Used time to translate a target. Unit: s.
    :return: ITR. Unit: bits/min.
    """
    if p < 0 or 1 < p:
        raise Exception('Accuracy need to be between 0 and 1.')
    elif p < 1 / n:
        warnings.warn('The ITR might be incorrect because the accuracy < chance level.')
        return 0
    elif p == 1:
        return math.log2(n) * 60 / t
    else:
        return (math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))) * 60 / t

def acc(y_true, y_pred):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    return np.sum(y_pred == y_true).astype(np.float64) / len(y_pred)


def batch_iter(data, batchsize, shuffle=True):
    data = np.array(list(data))
    data_size = data.shape[0]
    num_batches = np.ceil(data_size/batchsize).astype(np.int)
    # Shuffle the data
    if shuffle:
        shuffle_indices = shuffle_data(data_size)
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches):
        start_index = batch_num * batchsize
        end_index = min((batch_num + 1) * batchsize, data_size)
        yield shuffled_data[start_index:end_index]


def get_split_indices(data_size, split=[9, 1], shuffle=True):
    if len(split) < 2:
        raise TypeError('The length of split should be larger than 2 while the length of your split is {}!'.format(len(split)))
    split = np.array(split)
    split = split / np.sum(split)
    if shuffle:
        indices = shuffle_data(data_size)
    else:
        indices = np.arange(data_size)
    split_indices_list = []
    start = 0
    for i in range(len(split)-1):
        end = start + int(np.floor(split[i] * data_size))
        split_indices_list.append(indices[start:end])
        start = end
    split_indices_list.append(indices[start:])
    return split_indices_list


def shuffle_data(data_size, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    indices = np.arange(data_size)
    return np.random.permutation(indices).squeeze()
