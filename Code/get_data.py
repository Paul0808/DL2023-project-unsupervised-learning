import os
import pickle
import numpy as np
from PIL import Image

from utils import encode

from sklearn.model_selection import train_test_split


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_training_data(dataset="stl10"):
    if dataset == "stl10":
        with open("data/stl10_binary/train_X.bin", 'rb') as f:
            labeled_data = np.fromfile(f, dtype=np.uint8)
        with open("data/stl10_binary/train_y.bin", 'rb') as f:
            training_labels = np.fromfile(f, dtype=np.uint8)
        with open("data/stl10_binary/unlabeled_X.bin", 'rb') as f:
            unlabeled_data = np.fromfile(f, dtype=np.uint8)
        return labeled_data, training_labels, unlabeled_data
    elif dataset == "cifar10":
        batch1 = unpickle("data/cifar-10-batches-py/data_batch_1")
        batch2 = unpickle("data/cifar-10-batches-py/data_batch_2")
        batch3 = unpickle("data/cifar-10-batches-py/data_batch_3")
        batch4 = unpickle("data/cifar-10-batches-py/data_batch_4")
        batch5 = unpickle("data/cifar-10-batches-py/data_batch_5")
        meta = unpickle("data/cifar-10-batches-py/batches.meta")
        return batch1, batch2, batch3, batch4, batch5

def get_normalize_info(dataset="stl10"):
    if dataset == "stl10":
        train_data, _, _ = get_training_data(dataset=dataset)
        mean = np.mean(train_data)
        std = np.sqrt(np.std(train_data))
    elif dataset == "cifar10":
        batch1, batch2, batch3, batch4, batch5 = get_training_data(dataset=dataset)
        train_data = [*batch1[b'data'].astype('float32'), 
                    *batch2[b'data'].astype('float32'), 
                    *batch3[b'data'].astype('float32'), 
                    *batch4[b'data'].astype('float32'), 
                    *batch5[b'data'].astype('float32')]
        mean = np.mean([np.mean(img) for img in train_data])
        # TODO: Ask if we should use sqrt of std instead of just mean
        std = np.sqrt(np.mean([np.std(img) for img in train_data]))

    return mean,std



def get_test_data(dataset="stl10"):
    if dataset == "stl10":
        with open("data/stl10_binary/test_X.bin", 'rb') as f:
            test_data = np.fromfile(f, dtype=np.uint8)
        with open("data/stl10_binary/test_y.bin", 'rb') as f:
            test_labels = np.fromfile(f, dtype=np.uint8)
    elif dataset == "cifar10":
        test = unpickle("data/cifar-10-batches-py/test_batch")
        test_labels = [*test[b'labels']]
        test_data = [*test[b'data'].astype('float32')]

    if dataset == "stl10":
        test_data = np.reshape(test_data, (-1, 3, 96, 96))
        test_data = np.transpose(test_data, (0, 3, 2, 1)).astype('float32')
    elif dataset == "cifar10":
        test_data = np.array(test_data).reshape(len(test_data),3,32,32).transpose(0,2,3,1)
    # Normalize data
    # mean,std = get_normalize_info(dataset=dataset)
    # test_data -= mean
    # test_data /= std

    # One hot encoding test labels
    test_labels = encode(labels=test_labels, max_length=10)

    return test_data, test_labels

def get_unsupervised_data(data_type= "train", dataset="stl10"):
    if dataset == "cifar10":
        if data_type == "train":
            batch1, batch2, batch3, batch4, _ = get_training_data(dataset=dataset)
            data = [*batch1[b'data'].astype('float32'),
                    *batch2[b'data'].astype('float32'),
                    *batch3[b'data'].astype('float32'),
                    *batch4[b'data'].astype('float32')]
        elif data_type == "validation":
            _, _, _, _, batch5 = get_training_data(dataset=dataset)
            data = [*batch5[b'data'].astype('float32')]

    if (data_type == "train") or (data_type == "validation"):
        if dataset == "cifar10":
            data = np.array(data).reshape(len(data),3,32,32).transpose(0,2,3,1)
        
        elif dataset == "stl10":
            labeled_data, _, unlabeled_data = get_training_data(dataset=dataset)
            data = np.concatenate((labeled_data,unlabeled_data))
            data = np.reshape(data, (-1, 3, 96, 96))
            data = np.transpose(data, (0, 3, 2, 1)).astype('float32')

            if data_type == "train":
                data, _ = train_test_split(data, test_size=0.2, random_state=35)
            elif data_type == "validation":
                _, data = train_test_split(data, test_size=0.2, random_state=35)

        # Normalize data
        # mean,std = get_normalize_info(dataset=dataset)
        # data -= mean
        # data /= std

    elif data_type == "test":
        data, _ = get_test_data(dataset=dataset)
        # mean,std = get_normalize_info(dataset=dataset)
        # data -= mean
        # data /= std
    else:
        print(data_type)
        raise Exception("Not correct data type, choose: train, test or validation as input")
    
    # img = Image.fromarray(data[5].astype("uint8"))
    # img.show()
    return data

def get_semisupervised_data(dataset="stl10", no_batches:int = 1):
    if dataset == "stl10":
        labeled_data, training_labels, unlabeled_data = get_training_data(dataset=dataset)
    elif dataset == "cifar10":
        batch1, batch2, batch3, batch4, batch5 = get_training_data(dataset=dataset)
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

    if dataset == "stl10":
        unlabeled_data = np.reshape(unlabeled_data, (-1, 3, 96, 96))
        unlabeled_data = np.transpose(unlabeled_data, (0, 3, 2, 1)).astype('float32')
        labeled_data = np.reshape(labeled_data, (-1, 3, 96, 96))
        labeled_data = np.transpose(labeled_data, (0, 3, 2, 1)).astype('float32')
        labeled_data, validation_data, training_labels, validation_labels = train_test_split(labeled_data, training_labels, test_size=0.2, random_state=35)
        
    elif dataset == "cifar10":
        unlabeled_data = np.array(unlabeled_data).reshape(len(unlabeled_data),3,32,32).transpose(0,2,3,1)
        labeled_data = np.array(labeled_data).reshape(len(labeled_data),3,32,32).transpose(0,2,3,1)
        validation_data = np.array(validation_data).reshape(len(validation_data),3,32,32).transpose(0,2,3,1)
    

    # Normalize data
    # img = Image.fromarray(validation_data[5].astype("uint8"))
    # img.show()
    # mean,std = get_normalize_info(dataset=dataset)
    # unlabeled_data -= mean
    # unlabeled_data /= std
    # labeled_data -= mean
    # labeled_data /= std
    # validation_data -= mean
    # validation_data /= std
    # img = Image.fromarray(validation_data[5].astype("uint8"))
    # img.show()

    training_labels = encode(labels=training_labels, max_length=10)
    validation_labels = encode(labels=validation_labels, max_length=10)

    return labeled_data, training_labels, validation_data, validation_labels, unlabeled_data

def test():
    print(1)
    #labeled_data, training_labels, unlabeled_data = get_training_data()
    #print(training_labels.shape)
    # data = get_unsupervised_data()
    # labeled_data, training_labels, validation_data, validation_labels, unlabeled_data = get_semisupervised_data()
    # img = Image.fromarray(validation_data[3].astype("uint8"))
    # img.show()
    # data = get_unsupervised_data()
    # test, test_labels =get_test_data()
    # labeled_data, training_labels, validation_data, validation_labels, unlabeled_data = get_semisupervised_data()
    # print(labeled_data.shape)
    # print(get_normalize_info())
    # print('Data Type: %s' % labeled_data[0].dtype)
    # print('Min: %.3f, Max: %.3f' % (test[0].min(), test[0].max()))
    # print(test.shape)
    # print('Min: %.3f, Max: %.3f' % (labeled_data[0].min(), labeled_data[0].max()))
    # print(len(training_labels),labeled_data[0],labeled_data[0].shape, len(unlabeled_data), labeled_data[0].shape)
    

if __name__ == "__main__":
    test()