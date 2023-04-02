import os
import pickle
import numpy as np
from PIL import Image

from utils import encode


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
    # TODO: Ask if we should use sqrt of std instead of just mean
    std = np.sqrt(np.mean([np.std(img) for img in train_data]))
    return mean,std



def get_test_data():
    test = unpickle("data/cifar-10-batches-py/test_batch")
    test_labels = [*test[b'labels']]
    test_data = [*test[b'data'].astype('float32')]
    # Normalize data
    mean,std = get_normalize_info()
    test_data -= mean
    test_data /= std
    test_data = test_data.reshape(len(test_data),3,32,32).transpose(0,2,3,1)

    # One hot encoding test labels
    test_labels = encode(labels=test_labels, max_length=10)

    return test_data, test_labels

def get_unsupervised_data(data_type= "train"):
    if data_type == "train":
        batch1, batch2, batch3, batch4, _ = get_training_data()
        data = [*batch1[b'data'].astype('float32'),
                *batch2[b'data'].astype('float32'),
                *batch3[b'data'].astype('float32'),
                *batch4[b'data'].astype('float32')]
    elif data_type == "validation":
        _, _, _, _, batch5 = get_training_data()
        data = [*batch5[b'data'].astype('float32')]
    elif data_type == "test":
        _, data = get_test_data()
    else:
        print(data_type)
        raise Exception("Not correct data type, choose: train, test or validation as input")
    
    if (data_type == "train") or (data_type == "validation"):
        # Normalize data
        mean,std = get_normalize_info()
        data -= mean
        data /= std
        data = data.reshape(len(data),3,32,32).transpose(0,2,3,1)

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
    validation_labels=[]
    validation_data=[]
    unlabeled_data=[]
    
    for i in range(no_batches):
        training_labels = [*training_labels, *all_training_labels[i]]
        labeled_data = [*labeled_data, *all_training_data[i]]
    
    validation_labels = [*all_training_labels[no_batches]]
    validation_data = [*all_training_data[no_batches]]

    for i in range(no_batches + 1,5):
        unlabeled_data = [*unlabeled_data, *all_training_data[i]]

    # Normalize data
    mean,std = get_normalize_info()
    unlabeled_data -= mean
    unlabeled_data /= std
    unlabeled_data = unlabeled_data.reshape(len(unlabeled_data),3,32,32).transpose(0,2,3,1)
    labeled_data -= mean
    labeled_data /= std
    labeled_data = labeled_data.reshape(len(labeled_data),3,32,32).transpose(0,2,3,1)

    training_labels = encode(labels=training_labels, max_length=10)
    validation_labels = encode(labels=validation_labels, max_length=10)

    return labeled_data, training_labels, validation_data, validation_labels, unlabeled_data

def test():
    data = get_unsupervised_data()
    test, test_labels =get_test_data()
    labeled_data, training_labels, validation_data, validation_labels, unlabeled_data = get_semisupervised_data()
    print(labeled_data.shape)
    print(get_normalize_info())
    print('Data Type: %s' % labeled_data[0].dtype)
    print('Min: %.3f, Max: %.3f' % (test[0].min(), test[0].max()))
    print(test.shape)
    print('Min: %.3f, Max: %.3f' % (labeled_data[0].min(), labeled_data[0].max()))
    print(len(training_labels),labeled_data[0],labeled_data[0].shape, len(unlabeled_data), labeled_data[0].shape)
    

if __name__ == "__main__":
    test()