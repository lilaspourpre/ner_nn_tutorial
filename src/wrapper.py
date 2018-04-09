# -*- coding: utf-8 -*-
import numpy as np


def complement_data(x):
    array_x = np.array(x)
    seq_max_len = max([len(seq) for seq in x])
    #print('seq_max_len = ', seq_max_len)
    result_x = []
    for i in range(len(array_x)):
        li = [[0.] * int(np.array(array_x[i]).shape[1])] * int(seq_max_len - len(array_x[i]))
        result_x.append(np.concatenate((np.array(array_x[i]), np.array(li)), axis=0)) if li != [] else result_x.append(
            array_x[i])
    return result_x


def format_data(list_of_vectors, division):
    splitted_vectors = [[list_of_vectors.pop(0) for i in range(step)] for step in division]
    return splitted_vectors, [len(i) for i in splitted_vectors]
