
from keras.models import Model, load_model
from keras.layers import Input, Concatenate, Dense, Dropout
from keras import backend
from keras.utils import plot_model

import pickle as pkl
from numpy import concatenate

from resnet_backbone import resnet_creator
from utils import get_compiler_parameters
from jigsaw_creator import read_jigsaw_data
from get_data import get_semisupervised_data, get_test_data
from os import mkdir, environ
from pathlib import Path

# TODO: remove to run on gpu
environ["CUDA_VISIBLE_DEVICES"] = "-1"
class CFN(Model):
    def __init__(self, dataset="cifar10", permutations_no=100, crop_dimensions=27, crops_no=9, channels_no=3, resnet_architecture=[1, 1, 1], *args, **kwargs):
        self.dataset = dataset
        self.permutations_no = permutations_no
        self.crop_dimensions = crop_dimensions
        self.crops_no = crops_no
        self.channels_no = channels_no
        self.resnet_architecture = resnet_architecture

        # Declare the input for the 9 siamese blocks
        inputs= [Input((self.crop_dimensions, self.crop_dimensions, self.channels_no)) for _ in range(self.crops_no)]
        
        # Generating the 9 resnet blocks
        siammese_blocks = [resnet_creator(block_input, input_shape= (self.crop_dimensions, self.crop_dimensions, self.channels_no), architecture= self.resnet_architecture) for block_input in inputs]
        x = Concatenate()(siammese_blocks)

        # Declaring the first dense layer after the concatentation
        # One unit for each neuron in last layer so 256*9 (same as paper)
        x = Dense(units=2304, activation="relu")(x)
        x = Dropout(0.5)(x)

        # Declaring second dense layer after concatenation
        # Same units as paper
        x = Dense(units=2048, activation="relu")(x)
        x = Dropout(0.5)(x)

        # Declaring output layer
        x = Dense(units=self.permutations_no, activation="softmax")(x)

        super().__init__(inputs=inputs, outputs=x, *args, **kwargs)

    def train(self, epochs=100, batch_size=256, permutations_no=100):
        optimizer, callbacks = get_compiler_parameters()
        self.compile(optimizer= optimizer, metrics= ["accuracy"], loss= "categorical_crossentropy")
        
        x_train, y_train = read_jigsaw_data(data_type="train", dataset=self.dataset)
        x_validation, y_validation = read_jigsaw_data(data_type="validation", dataset=self.dataset)
        
        history = self.fit(x= x_train, y= y_train, epochs= epochs, batch_size=batch_size, validation_data= (x_validation, y_validation), callbacks= callbacks, shuffle= False)
        
        self.save("models/CFN_" + self.dataset[:-2] + "_" + str(permutations_no) + ".hdf5")

        if not(Path("models/history").exists()):
            mkdir("models/history")

        with open("models/history/history_unsupervised_" + self.dataset[:-2] + ".pkl", 'wb') as history_file:
            pkl.dump(history.history, history_file)
        
        return history
    
    def test(self):
        x_test, y_test = read_jigsaw_data("test", dataset=self.dataset)
        scores = self.evaluate(x= x_test, y= y_test)
        
        if not(Path("models/scores").exists()):
            mkdir("models/scores")
        with open("models/history/scores_unsupervised_" + self.dataset[:-2] + ".pkl", 'wb') as scores_file:
            pkl.dump(scores, scores_file)

        print("Test loss: " + str(scores[0]))
        print("Test accuracy: " + str(scores[1]))
    
    def get_config(self):
        config = super().get_config()
        config.update({"name": self.name,
        "dataset": self.dataset,
        "permutations_no": self.permutations_no,
        "crop_dimensions": self.crop_dimensions,
        "crops_no": self.crops_no,
        "channels_no": self.channels_no,
        "resnet_architecture": self.resnet_architecture})
        return config

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

class CFN_transfer(Model):
    def __init__(self,dataset="cifar10", classes_no=10, image_dimensions=96, siamese_no=9, channels_no=3, resnet_architecture=[1, 1, 1], *args, **kwargs):
        self.dataset = dataset
        self.classes_no= classes_no
        self.image_dimensions= image_dimensions
        self.siamese_no= siamese_no
        self.channels_no= channels_no
        self.resnet_architecture= resnet_architecture
        # Declare the input for the 1 image 32x32x3
        inputs= Input((self.image_dimensions, self.image_dimensions, self.channels_no))
        
        # Generating the 9 resnet blocks
        siammese_blocks = [resnet_creator(inputs, input_shape= (self.image_dimensions, self.image_dimensions, self.channels_no), architecture= self.resnet_architecture) for _ in range(self.siamese_no)]
        
        x = Concatenate()(siammese_blocks)

        # Declaring the first dense layer after the concatentation
        # One unit for each neuron in last layer so 256*9 (same as paper)
        x = Dense(units=2304, activation="relu")(x)
        x = Dropout(0.5)(x)

        # Declaring second dense layer after concatenation
        # Same units as paper
        x = Dense(units=2048, activation="relu")(x)
        x = Dropout(0.5)(x)

        # Declaring output layer
        x = Dense(units=self.classes_no, activation="softmax")(x)

        super().__init__(inputs=inputs, outputs=x, *args, **kwargs)

    # TODO: change train to unsupervised data
    def train(self, epochs=100, batch_size=64, permutations_no= 100):
        optimizer, callbacks = get_compiler_parameters()
        self.compile(optimizer= optimizer, metrics= ["accuracy"], loss= "categorical_crossentropy")
        
        x_train, y_train, x_validation, y_validation, _ = get_semisupervised_data(dataset=self.dataset)
        
        history = self.fit(x= x_train, y= y_train, epochs= epochs, batch_size= batch_size, validation_data= (x_validation, y_validation), callbacks= callbacks)
        
        self.save("models/CFN_" + self.dataset[:-2] + "_labeled.hdf5")

        if not(Path("models/history").exists()):
            mkdir("models/history")

        with open("models/history/history_CFN_" + self.dataset[:-2] + "_labeled.pkl", 'wb') as history_file:
            pkl.dump(history.history, history_file)
        return history
    
    def test(self, data_type="labeled"):
        x_test, y_test = get_test_data(dataset=self.dataset)
        scores = self.evaluate(x= x_test, y= y_test)
        
        if not(Path("models/scores").exists()):
            mkdir("models/scores")
        with open("models/scores/scores_CFN_" + self.dataset[:-2] + "_" + data_type + ".pkl", 'wb') as scores_file:
            pkl.dump(scores, scores_file)

        print("Test loss: " + str(scores[0]))
        print("Test accuracy: " + str(scores[1]))

    def get_config(self):
        config = super().get_config()
        config.update({"name": self.name,
        "dataset": self.dataset,
        "classes_no": self.classes_no,
        "image_dimensions": self.image_dimensions,
        "siamese_no": self.siamese_no,
        "channels_no": self.channels_no,
        "resnet_architecture": self.resnet_architecture})
        return config

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class ResNet34_refrence(Model):
    def __init__(self, dataset="cifar10", classes_no=10, image_dimensions=32, channels_no=3, resnet_architecture=[3, 4, 6, 3], *args, **kwargs):
        self.dataset = dataset
        self.classes_no= classes_no
        self.image_dimensions= image_dimensions
        self.channels_no= channels_no
        self.resnet_architecture= resnet_architecture
        # Declare the input for the 1 image 32x32x3
        inputs= Input((self.image_dimensions, self.image_dimensions, self.channels_no))
        
        # Declaring 1 renset block
        resnet = resnet_creator(inputs, input_shape= (self.image_dimensions, self.image_dimensions, self.channels_no), architecture= self.resnet_architecture)
        
        # Declaring output layer
        resnet = Dense(units=self.classes_no, activation="softmax")(resnet)

        super().__init__(inputs=inputs, outputs=resnet, *args, **kwargs)

    # TODO: change train to unsupervised data
    def train(self, epochs=100, batch_size=64, data_type="labeled"):
        optimizer, callbacks = get_compiler_parameters()
        self.compile(optimizer= optimizer, metrics= ["accuracy"], loss= "categorical_crossentropy")
        
        if data_type == "labeled":
            x_train, y_train, x_validation, y_validation, _ = get_semisupervised_data(dataset=self.dataset)
            history = self.fit(x= x_train, y= y_train, epochs= epochs, batch_size= batch_size, validation_data= (x_validation, y_validation), callbacks= callbacks)
            
        elif data_type == "full":
            x_train, y_train, x_validation, y_validation, unlabeled = get_semisupervised_data(dataset=self.dataset)
            history = self.fit(x= x_train, y= y_train, epochs= epochs, batch_size= batch_size, validation_data= (x_validation, y_validation), callbacks= callbacks)
            
            x_train= concatenate((x_train, unlabeled), axis=0)
            y_unlabeled = self.predict(unlabeled)
            y_train= concatenate((y_train, y_unlabeled), axis=0)

            history = self.fit(x= x_train, y= y_train, epochs= epochs, batch_size= batch_size, validation_data= (x_validation, y_validation), callbacks= callbacks)

        self.save("models/Resnet34_" + self.dataset[:-2] + "_" + data_type + ".hdf5")
        
        if not(Path("models/history").exists()):
            mkdir("models/history")

        with open("models/history/history_reference_" + self.dataset[:-2] + "_" + data_type + ".pkl", 'wb') as history_file:
            pkl.dump(history.history, history_file)
        
        return history
    
    def test(self, data_type="labeled"):
        x_test, y_test = get_test_data(dataset=self.dataset)
        scores = self.evaluate(x= x_test, y= y_test)
        
        if not(Path("models/scores").exists()):
            mkdir("models/scores")
        with open("models/scores/scores_reference_"+ self.dataset[:-2] + "_" + data_type + ".pkl", 'wb') as scores_file:
            pkl.dump(scores, scores_file)

        print("Test loss refrence: " + str(scores[0]))
        print("Test accuracy refrence: " + str(scores[1]))
    
    def get_config(self):
        config = super().get_config()
        config.update({"name": self.name,
        "dataset": self.dataset,
        "classes_no": self.classes_no,
        "image_dimensions": self.image_dimensions,
        "channels_no": self.channels_no,
        "resnet_architecture": self.resnet_architecture})
        return config

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)



# model = CFN_transfer()
# plot_model(model, to_file="models/CFN_multiple_inputs.png")
# model.summary()
# model.train()
# model.test()
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