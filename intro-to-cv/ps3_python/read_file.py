import numpy as np


def read_file(fname):
    data = []

    with open(fname, 'r') as f:
        for line in f:
            tokens = list(map(float, line.strip().split()))
            data.append(tokens)

    return np.array(data)
