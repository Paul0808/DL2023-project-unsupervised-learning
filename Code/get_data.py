import os
import pickle
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_training_data():
    batch1 = unpickle("data/cifar-10-batches-py/data_batch_1")
    batch2 = unpickle("data/cifar-10-batches-py/data_batch_2")
    batch3 = unpickle("data/cifar-10-batches-py/data_batch_3")
    batch4 = unpickle("data/cifar-10-batches-py/data_batch_4")
    batch5 = unpickle("data/cifar-10-batches-py/data_batch_5")
    meta = unpickle("data/cifar-10-batches-py/batches.meta")
    return batch1, batch2, batch3, batch4, batch5

def get_normalize_info():
    batch1, batch2, batch3, batch4, batch5 = get_training_data()
    train_data = [*batch1[b'data'].astype('float32'), 
                  *batch2[b'data'].astype('float32'), 
                  *batch3[b'data'].astype('float32'), 
                  *batch4[b'data'].astype('float32'), 
                  *batch5[b'data'].astype('float32')]
    mean = np.mean([np.mean(img) for img in train_data])
    std = np.std([np.std(img) for img in train_data])
    return mean,std



def get_test_data():
    test = unpickle("data/cifar-10-batches-py/test_batch")
    test_labels = [*test[b'labels']]
    test_data = [*test[b'data']]
    # Normalize data
    mean,std = get_normalize_info()
    test_data -= mean
    test_data /= std
    return test_labels, test_data

def get_unsupervised_data():
    batch1, batch2, batch3, batch4, batch5 = get_training_data()
    data = [*batch1[b'data'].astype('float32'),
            *batch2[b'data'].astype('float32'),
            *batch3[b'data'].astype('float32'),
            *batch4[b'data'].astype('float32'),
            *batch5[b'data'].astype('float32')]
    # Normalize data
    mean,std = get_normalize_info()
    data -= mean
    data /= std
    return data

def get_semisupervised_data(no_batches:int = 1):
    batch1, batch2, batch3, batch4, batch5 = get_training_data()
    all_training_labels = [batch1[b'labels'], batch2[b'labels'], batch3[b'labels'], batch4[b'labels'], batch5[b'labels']]
    all_training_data = [batch1[b'data'].astype('float32'), 
                         batch2[b'data'].astype('float32'), 
                         batch3[b'data'].astype('float32'), 
                         batch4[b'data'].astype('float32'), 
                         batch5[b'data'].astype('float32')]
    
    training_labels=[]
    labeled_data=[]
    unlabeled_data=[]
    
    for i in range(no_batches):
        training_labels = [*training_labels, *all_training_labels[i]]
        labeled_data = [*labeled_data, *all_training_data[i]]

    for i in range(no_batches,5):
        unlabeled_data = [*unlabeled_data, *all_training_data[i]]

    # Normalize data
    mean,std = get_normalize_info()
    unlabeled_data -= mean
    unlabeled_data /= std
    labeled_data -= mean
    labeled_data /= std

    return training_labels, labeled_data, unlabeled_data
def test():
    data = get_unsupervised_data()
    _,test =get_test_data()
    training_labels, labeled_data, unlabeled_data = get_semisupervised_data()
    get_normalize_info()
    print('Data Type: %s' % labeled_data[0].dtype)
    print('Min: %.3f, Max: %.3f' % (data[0].min(), data[0].max()))
    print('Min: %.3f, Max: %.3f' % (test[0].min(), test[0].max()))
    print(len(training_labels),labeled_data, len(unlabeled_data))

if __name__ == "__main__":
    test()