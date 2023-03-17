import os
import pickle


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

batch1 = unpickle("data/cifar-10-batches-py/data_batch_1")
meta = unpickle("data/cifar-10-batches-py/batches.meta")

print(batch1[b'labels'])
print(meta)