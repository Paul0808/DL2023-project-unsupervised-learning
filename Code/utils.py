
from itertools import permutations

import numpy as np
from numpy.random import randint

from scipy.spatial.distance import cdist

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from time import strftime, localtime


import h5py
from os import mkdir
from pathlib import Path

def generate_hamming(crops_no = 9, permutations_no = 100):

    initial_index_vector = np.array([*range(crops_no)])
    all_permutations = np.array([*permutations(initial_index_vector, crops_no)])

    # Assign random permutation index
    index = randint(len(all_permutations))

    # Assign that permutation to the final permutation list
    final_permutations = all_permutations[index].reshape((1, -1))

    # Repeat process for the left amount of permutations needed
    for _ in range(1, permutations_no):
        # Delete the previously assigned permutation for the all permutations array
        all_permutations = np.delete(all_permutations, index, axis=0)
        
        # Compute all mean hamming distances between all left permutations and final desigred permutations vector
        hamming_dist = cdist(final_permutations, all_permutations, metric='hamming').mean(axis=0).flatten()

        # Update the index based on the index of the maximum computed Hamming distance
        index = hamming_dist.argmax()

        # Add the permutation relating to the highest hamming distance to the final "desired" permutaions vector
        final_permutations = np.concatenate((final_permutations, all_permutations[index].reshape((1, -1))), axis=0)
    
    # Save constructed hamming set
    if not(Path("data/unsupervised").exists()):
        mkdir("data/unsupervised")
    if not(Path("data/unsupervised/hamming").exists()):
        mkdir("data/unsupervised/hamming")
    file = h5py.File("data/unsupervised/hamming/hamming_" + str(permutations_no) + '.h5', 'w')
    file.create_dataset('hamming', data= final_permutations)
    file.close()
    print("Hamming distance set has been saved for: " + str(permutations_no))

    # Returning the final permutations
    return final_permutations

# TODO: might need changing
def get_compiler_parameters(optimizer_lr=0.001, optimizer_beta_1=0.9, optimizer_beta_2=0.999,
                            stopping_monitor="val_loss", stopping_patience=10, stopping_verbose=1, 
                            checkpoint_monitor="val_loss", checkpoint_verbose=1, 
                            plateau_monitor="val_loss", plateau_patience=3, plateau_verbose=1):

    optimizer = Adam(learning_rate= optimizer_lr, beta_1= optimizer_beta_1, beta_2= optimizer_beta_2)

    stop = EarlyStopping(monitor= stopping_monitor, patience= stopping_patience, verbose= stopping_verbose)
    
    if not(Path("models").exists()):
        mkdir("models")
    if not(Path("models/checkpoints").exists()):
        mkdir("models/checkpoints")
    path_checkpoint = "models/checkpoints/{}".format(strftime("%b_%d_%H_%M_%S", localtime()))
    mkdir(path_checkpoint)

    checkpoint = ModelCheckpoint(path_checkpoint + "/weights.hdf5", monitor= checkpoint_monitor, verbose= checkpoint_verbose, save_best_only= True)
    
    plateau_reduction = ReduceLROnPlateau(monitor= plateau_monitor, patience= plateau_patience, verbose= plateau_verbose)

    callbacks = [stop, checkpoint, plateau_reduction]
    return optimizer, callbacks


# Function that one hot encodes the labels
def encode(labels, max_length):
    labels = np.array([[0 if labels[i] != j else 1 for j in range(max_length)] for i in range(len(labels))])
    return labels