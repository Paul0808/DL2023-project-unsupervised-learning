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

def get_test_data():
    test = unpickle("data/cifar-10-batches-py/test_batch")
    test_labels = [*test[b'labels']]
    test_data = [*test[b'data']]
    return test_labels, test_data

def get_unsupervised_data():
    batch1, batch2, batch3, batch4, batch5 = get_training_data()
    data = [*batch1[b'data'], *batch2[b'data'], *batch3[b'data'], *batch4[b'data'], *batch5[b'data']]
    return data

def get_semisupervised_data(no_batches:int = 1):
    batch1, batch2, batch3, batch4, batch5 = get_training_data()
    all_training_labels = [batch1[b'labels'], batch2[b'labels'], batch3[b'labels'], batch4[b'labels'], batch5[b'labels']]
    all_training_data = [batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data']]
    
    training_labels=[]
    labeled_data=[]
    unlabeled_data=[]
    
    for i in range(no_batches):
        training_labels = [*training_labels, *all_training_labels[i]]
        labeled_data = [*labeled_data, *all_training_data[i]]

    for i in range(no_batches,5):
        unlabeled_data = [*unlabeled_data, *all_training_data[i]]

    return training_labels, labeled_data, unlabeled_data

def get_mean_stdev_training_data():
    batch1, batch2, batch3, batch4, batch5 = get_training_data()
    mean_list = []
    std_list = []

    for i in batch1[b'data']:
        mean_list.append(np.mean(i))
        std_list.append(np.std(i))

    for i in batch2[b'data']:
        mean_list.append(np.mean(i))
        std_list.append(np.std(i))

    for i in batch3[b'data']:
        mean_list.append(np.mean(i))
        std_list.append(np.std(i))

    for i in batch4[b'data']:
        mean_list.append(np.mean(i))
        std_list.append(np.std(i))

    for i in batch5[b'data']:
        mean_list.append(np.mean(i))
        std_list.append(np.std(i))


    mean_training = np.mean(mean_list)
    std_training = np.std(std_list)

    return mean_training, std_training
    


def normalize_semisupervised_data():
    _, labeled_data, unlabeled_data = get_semisupervised_data()

    mean_training, std_training = get_mean_stdev_training_data()

    for i in range(len(labeled_data)):
        labeled_data[i] = [(float(j) - mean_training) / std_training for j in labeled_data[i]]

    for i in range(len(unlabeled_data)):
        unlabeled_data[i] = [(float(j) - mean_training) / std_training for j in unlabeled_data[i]]

    return labeled_data, unlabeled_data

def normalize_unsupervised_data():
    unsupervised_data = get_unsupervised_data()
    #unsupervised_data_normalized = [float(i) / sum(unsupervised_data) for i in unsupervised_data]

    mean_training, std_training = get_mean_stdev_training_data()

    for i in range(len(unsupervised_data)):
        unsupervised_data[i] = [(float(j) - mean_training) / std_training for j in unsupervised_data[i]]

    return unsupervised_data

def normalize_test_data():
    _, test_data = get_test_data()

    mean_training, std_training = get_mean_stdev_training_data()

    for i in len(test_data):
        test_data[i] = [(float(j) - mean_training) / std_training for j in test_data[i]]

    return test_data


labeled_data, unlabeled_data = normalize_semisupervised_data()

print(labeled_data)
