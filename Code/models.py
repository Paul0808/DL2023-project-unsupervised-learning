
from keras.models import Model
from keras.layers import Input, Concatenate, Dense, Dropout
from keras import backend

import pickle as pkl

from resnet_backbone import resnet_creator
from utils import get_compiler_parameters
from jigsaw_creator import read_jigsaw_data
from get_data import get_semisupervised_data, get_test_data
from os import mkdir, environ
from pathlib import Path

# TODO: remove to run on gpu
environ["CUDA_VISIBLE_DEVICES"] = "-1"
class CFN(Model):
    def __init__(self, permutations_no=100, crop_dimensions=7, crops_no=9, channels_no=3, resnet_architecture=[1, 1, 1], *args, **kwargs):
        self.resnet_architecture = resnet_architecture
        # Declare the input for the 9 siamese blocks
        inputs= [Input((crop_dimensions, crop_dimensions, channels_no)) for _ in range(crops_no)]
        
        # Generating the 9 resnet blocks
        resnet = resnet_creator(inputs[0], input_shape= (crop_dimensions, crop_dimensions, channels_no), architecture= resnet_architecture)
        siammese_blocks = [resnet_creator(block_input, input_shape= (crop_dimensions, crop_dimensions, channels_no), architecture= resnet_architecture) for block_input in inputs]

        x = Concatenate()(siammese_blocks)

        # Declaring the first dense layer after the concatentation
        # One unit for each neuron in last layer so 512*9 (same as paper)
        x = Dense(units=4608, activation="relu")(x)
        x = Dropout(0.5)(x)

        # Declaring second dense layer after concatenation
        # Same units as paper
        x = Dense(units=4096, activation="relu")(x)
        x = Dropout(0.5)(x)

        # Declaring output layer
        x = Dense(units=permutations_no, activation="softmax")(x)

        super().__init__(inputs=inputs, outputs=x, name="CFN", *args, **kwargs)

    def train(self, epochs=100, batch_size=64, permutations_no=100):
        optimizer, callbacks = get_compiler_parameters()
        self.compile(optimizer= optimizer, metrics= ["accuracy"], loss= "categorical_crossentropy")
        
        x_train, y_train = read_jigsaw_data(data_type="train")
        x_validation, y_validation = read_jigsaw_data(data_type="validation")
        
        history = self.fit(x= x_train, y= y_train, epochs= epochs, batch_size=batch_size, validation_data= (x_validation, y_validation), callbacks= callbacks)
        
        self.save("models/CFN_" + str(permutations_no) + "_blockx" + str(self.resnet_architecture[0]) + "_len" + len(self.resnet_architecture) + ".hdf5")

        if not(Path("models/history").exists()):
            mkdir("models/history")

        with open("models/history/history_unsupervised" + "_blockx" + str(self.resnet_architecture[0]) + "_len" + len(self.resnet_architecture) + ".pkl", 'wb') as history_file:
            pkl.dump(history.history, history_file)
        
        return history
    
    def test(self):
        x_test, y_test = read_jigsaw_data("test")
        scores = self.evaluate(x= x_test, y= y_test)
        
        if not(Path("models/scores").exists()):
            mkdir("models/scores")
        with open("models/history/scores_unsupervised" + "_blockx" + str(self.resnet_architecture[0]) + "_len" + len(self.resnet_architecture) + ".pkl", 'wb') as scores_file:
            pkl.dump(scores, scores_file)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

class CFN_transfer(Model):
    def __init__(self, classes_no=10, image_dimensions=32, siamese_no=9, channels_no=3, resnet_architecture=[2, 2, 2, 2], *args, **kwargs):
        # Declare the input for the 1 image 32x32x3
        inputs= Input((image_dimensions, image_dimensions, channels_no))
        
        # Generating the 9 resnet blocks
        siammese_blocks = [resnet_creator(inputs, input_shape= (image_dimensions, image_dimensions, channels_no), architecture= resnet_architecture) for _ in range(siamese_no)]
        x = Concatenate()(siammese_blocks)

        # Declaring the first dense layer after the concatentation
        # One unit for each neuron in last layer so 512*9 (same as paper)
        x = Dense(units=4608, activation="relu")(x)
        x = Dropout(0.5)(x)

        # Declaring second dense layer after concatenation
        # Same units as paper
        x = Dense(units=4096, activation="relu")(x)
        x = Dropout(0.5)(x)

        # Declaring output layer
        x = Dense(units=classes_no, activation="softmax")(x)

        super().__init__(inputs=inputs, outputs=x, name="CFN_transfer", *args, **kwargs)

    # TODO: change train to unsupervised data
    def train(self, epochs=100, batch_size=64, permutations_no= 100):
        optimizer, callbacks = get_compiler_parameters()
        self.compile(optimizer= optimizer, metrics= ["accuracy"], loss= "categorical_crossentropy")
        
        x_train, y_train = read_jigsaw_data(data_type= "train", permutations_no= permutations_no)
        x_validation, y_validation = read_jigsaw_data(data_type= "validation", permutations_no= permutations_no)
        
        history = self.fit(x= x_train, y= y_train, epochs= epochs, batch_size= batch_size, validation_data= (x_validation, y_validation), callbacks= callbacks)
        
        self.save("models/CFN_transfered.hdf5")
        return history

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class ResNet34_refrence(Model):
    def __init__(self, classes_no=10, image_dimensions=32, channels_no=3, siamese_architecture=[3, 4, 6, 3], *args, **kwargs):
        # Declare the input for the 1 image 32x32x3
        inputs= Input((image_dimensions, image_dimensions, channels_no))
        
        # Declaring 1 renset block
        resnet = resnet_creator(inputs, input_shape= (image_dimensions, image_dimensions, channels_no), architecture= siamese_architecture)
        
        # Declaring output layer
        resnet = Dense(units=classes_no, activation="softmax")(resnet)

        super().__init__(inputs=inputs, outputs=resnet, name= "ResNet34", *args, **kwargs)

    # TODO: change train to unsupervised data
    def train(self, epochs=100, batch_size=64, data_type="labeled"):
        optimizer, callbacks = get_compiler_parameters()
        self.compile(optimizer= optimizer, metrics= ["accuracy"], loss= "categorical_crossentropy")
        
        x_train, y_train, x_validation, y_validation, _ = get_semisupervised_data()
        
        history = self.fit(x= x_train, y= y_train, epochs= epochs, batch_size= batch_size, validation_data= (x_validation, y_validation), callbacks= callbacks)
        self.save("models/resnet.hdf5")

        if not(Path("models/history").exists()):
            mkdir("models/history")

        with open("models/history/history_reference_" + data_type + ".pkl", 'wb') as history_file:
            pkl.dump(history.history, history_file)
        
        return history
    
    def test(self, data_type="labeled"):
        x_test, y_test = get_test_data()
        scores = self.evaluate(x= x_test, y= y_test)
        
        if not(Path("models/scores").exists()):
            mkdir("models/scores")
        with open("models/scores/scores_reference_" + data_type + ".pkl", 'wb') as scores_file:
            pkl.dump(scores, scores_file)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)



model = CFN()
model.summary()
model.train()
model.test()
# model1 = CFN_transfer()

#model.summary()
# for i in range(len(model1.layers)-1):
#     model1.layers[i].set_weights(model.layers[8+i].weights)
# model.compile(loss="categorical_crossentropy",metrics=["accuracy"])
# model.summary
# x,y = read_jigsaw_data()
# model.fit(x=x,y=y)

# #model1.summary
# del model
#model.outputs = [model.layers[-1].output]
#model.input = Input((32,32,3))
#x = Sequential([Input((32,32,3)), model])
#x.layers[0].weights
#x = model.layers[9:]
#model = Model(inputs=Input((32,32,3)), outputs= model.layers[9:-1].output)
#print(model.layers[:3])