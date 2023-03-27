from get_data import get_unsupervised_data
from utils import generate_hamming

from random import randrange, randint

import numpy as np

import h5py
from os import mkdir
from pathlib import Path

# Funtion that jitters the colour channels of the image to prevent colour aberation and overfitting
# By default it will jitter all colour channels, in all directions randomly within 1 pixel
# in original a jitter factor of 2 was used but cifar dataset has much lower quality images
def jitter(image, jitter_factor= 1):
    # The jitter area should have a border in all directions based on the jitter factor
    jitter_area = len(image) - jitter_factor*2

    # Randomly Generating the jitter factor for  all directions
    x_jitter = randrange(jitter_factor*2 + 1)
    y_jitter = randrange(jitter_factor*2 + 1)

    # Create an empty array with the same shape that the output image will have ((w-2)x(h-2)x3, where the w=h)
    jittered_image = np.empty((jitter_area, jitter_area, 3), np.float32)

    # Assign the jittered image each jittered channel
    for channel in range(3):
        jittered_image[:, :, channel] = image[x_jitter:(jitter_area + x_jitter), y_jitter:(jitter_area + y_jitter), channel]
    
    return jittered_image

def create_jigsaw(image, hamming, permutations_no= 100, crop_dimensions= 7, crops_no= 9, crop_area= 27):
    img_size = len(image)

    # Declaring the zone size within where a crop should be performed 
    # (3 crops in each direction so should be equal to crop area divided by 3)
    zone_size = crop_area/3 

    # Jitter colour to prevent colour aberation problem
    image = jitter(image)

    # Generate random coordinates for where the cropping area should start
    start_x = randrange(img_size-crop_area)
    start_y = randrange(img_size-crop_area)

    # Getting a random index to extract a permutation from the hamming set
    index = randrange(permutations_no)

    # Declare empty array 4d array with the dimensions of the output jigsaw puzzle
    # By default 7x7x3x9 (because there are 9 tiles and the crop dimension 7)
    jigsaw_puzzle = np.empty((crop_dimensions, crop_dimensions, 3, 9), np.float32)

    # Create all crops
    for row in range(int(np.sqrt(crops_no))):
        for column in range(int(np.sqrt(crops_no))):
            # Declare the coordinates of where the active crop zone starts
            zone_x = int((zone_size * column) + randrange(zone_size-crop_dimensions) + start_x)
            zone_y = int((zone_size * row) + randrange(zone_size-crop_dimensions) + start_y)
            
            # Add each cropping made to the puzzle based on the permutation selected
            jigsaw_puzzle[:, :, :, hamming[index, (3*row) + column]] = image[zone_y:(zone_y + crop_dimensions),
                                                                           zone_x:(zone_x + crop_dimensions), :]
    
    return jigsaw_puzzle, index

# Function that one hot encodes the labels
def encode(labels, max_length):
    labels = np.array([[0 if labels[i] != j else 1 for j in range(max_length)] for i in range(len(labels))])
    return labels

def generate_jigsaw_data(data_type= "train", permutations_no=100, crop_area= 27, crop_dimensions= 7, crops_no= 9, channels_no= 3):
    # Import the correct data for training validation and testing
    if data_type == "train":
        images= get_unsupervised_data("train")
    elif data_type == "validation":
        images= get_unsupervised_data("validation")
    elif data_type == "test":
        images= get_unsupervised_data("test")
    else:
        print(data_type)
        raise Exception("Not correct data type, choose: train, test or validation as input")

    # Create empty arrays for the data and labels, coresponding to the final dimension that they should have
    x = [np.empty((len(images), crop_dimensions, crop_dimensions, channels_no), np.float32)]*crops_no
    y = np.empty(len(images))
    
    # Getting the hamming set if available otherwise generate a hamming set
    try:
        file = h5py.File("data/unsupervised/hamming/hamming_" + str(permutations_no) + '.h5', 'r')
    except FileNotFoundError:
        generate_hamming(crops_no=crops_no, permutations_no=permutations_no)
        file = h5py.File("data/unsupervised/hamming/hamming_" + str(permutations_no) + '.h5', 'r')
    
    # Extracting hamming set from file
    hamming = np.array(file['hamming'])
    file.close
    
    # For each image create a jigsaw puzzle, with its label and append it to their specific arrays/list
    for index_img in range(len(images)):
        jigsaw, y[index_img] = create_jigsaw(images[index_img], hamming, permutations_no, crop_dimensions, crops_no, crop_area)
        
        for index_crop in range(crops_no):
            x[index_crop][index_img, :, :, :] = jigsaw[:, :, :, index_crop]

    # One hot encoding the labels
    y = encode(y, len(hamming))

    # Save jigsaw dataset
    if not(Path("data/unsupervised").exists()):
        mkdir("data/unsupervised")
    file = h5py.File("data/unsupervised/"+ str(data_type)+ "_" + str(permutations_no) + '.h5', 'w')
    file.create_dataset(str(data_type) + '_data', data= x)
    file.create_dataset(str(data_type) + '_labels', data= y)
    file.close()
    print("Jigsaw " + str(data_type) + " dataset has been saved for: " + str(permutations_no) + " permutations")


    # Returning the jigsaw data and labels
    return x,y

def read_jigsaw_data(data_type= "train", permutations_no=100):
    try:
        file = h5py.File("data/unsupervised/"+ str(data_type)+ "_" + str(permutations_no) + '.h5', 'r')
    except FileNotFoundError:
        generate_jigsaw_data(data_type=data_type, permutations_no=permutations_no)
        file = h5py.File("data/unsupervised/"+ str(data_type)+ "_" + str(permutations_no) + '.h5', 'r')
    x_saved = list(file['train_data'])
    y_saved = np.array(file["train_labels"])
    file.close

    return x_saved, y_saved
    

# x,y = generate_jigsaw_data()
# # Getting the hamming set if available otherwise generate a hamming set

# file = h5py.File("data/unsupervised/train_100.h5", 'r')

# # Extracting hamming set from file
# x_saved, y_saved = read_jigsaw_data()
# #x_saved = list(x_saved)
# print(x_saved[0].shape)