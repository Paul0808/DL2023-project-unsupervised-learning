# DL2023-project-unsupervised-learning
Final Project for Deep Learning

# Pre-requisites 

Download either the CIFAR10 dataset or the STL10 from their corresponding websites and unzip them in the "data" folder.

WARNING! Running the self-supervised pretext task on the stl10 dataset requires much computational resources

# Project Structure

The project contains a "run.py" script for running the code, a "requirements.txt" file and a "data" folder where the data necessary for assignment needs to be "Code" folder with all the aditional scripts necessary for running the code

- **run.py** : script that needs to be run to run each individual task

## Code Folder

Includes:
- **color_affine_layer.py** : script containing a keras layer class to be used in the custom keras models. Used in **models.py**

- **get_data.py** : script containing all the function necessary for reading the data for each dataset.py

- **jigsaw_creator.py** : script containing all the function necessary for transforming the datasets into the required jigsaw dataset.

- **models.py** : script including custom classes for the developed models + functions for training and testing models

- **resnet_backbone.py** : script including all functions needed to generate the ResNet Architecture

- **transfer_learning.py** : script including all functions to perform the transfer learning and training the model with transferred weights

- **utils.py** : script including utilitary functions


## data Folder

Should be included:
- **cifar-10-batches-py** : folder containing batch data for CIFAR10 dataset

- **stl10_binary** : folder containing batch data for STL10 dataset

## Requirements

h5py==3.7.0

keras==2.10.0

numpy==1.23.3

Pillow==9.5.0

scikit_learn==1.2.2

scipy==1.10.1

tensorflow==2.10.0

tqdm==4.64.1

## The Report
report is given as a pdf file

# Running Task

## Prerequisit:

pip install -r requirements.txt

## Running task

1. Either directly run the script "run.py" or use command "python run.py" from the working directory

2. To exclude different parts of the tasks use function call arguments:

**--task**: selecting what task to be performed (semi, self, class)
**--model**: choosing what model to be used for the simple classification (cfn_transfered, resnet)
**--dataset**: select dataset to be tested on (stl10, cifar10)
**--batch-size**: batch size to be performed (as an int)

# Output

Data split into labeled, unlabeled and test are also saved in this folder. In adition the dataset with the predicted labels from semi supervised learning is also saved here

## models Folder

All trained models are saved with coresponding names in this folder, history and scores of the models after training and testing are also saved here in their corresponding folders

# Copyright
Scripts created by:

Adrian Serbanescu (s3944735) && Andra Cristiana Minculescu (s3993507) && Rares Adrian Oancea (s3974537) && Paul Pintea (s3593673)