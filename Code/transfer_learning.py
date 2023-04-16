from keras.models import Model, load_model

from Code.models import CFN, CFN_transfer

from Code.get_data import get_semisupervised_data, get_test_data
from Code.utils import get_compiler_parameters

import pickle as pkl
from numpy import concatenate

from os import mkdir, environ
from pathlib import Path

def transfer_weights(dataset="cifar10"):
    if dataset == "cifar10":
        try:
            model = load_model("models/CFN_cifar_100.hdf5", custom_objects={"CFN": CFN})#custom_objects={"CustomModel": CustomModel}
        except FileNotFoundError:
            raise Exception("Model not found try trining the CFN in the self-supervised task for the " + dataset + " dataset")
        transfered_model = CFN_transfer(dataset=dataset,image_dimensions=32)
    elif dataset == "stl10":
        try:
            model = load_model("models/CFN_stl_100.hdf5", custom_objects={"CFN": CFN})
        except FileNotFoundError:
            raise Exception("Model not found try trining the CFN in the self-supervised task for the " + dataset + " dataset")
        transfered_model = CFN_transfer(dataset=dataset,image_dimensions=96)

    
    # Transfering the weights from the pretrained model to the new model, without the input/inputs 
    # and the last 105 layers from the pretrained This means that only the weights up to and including the second resnet block will be transfered
    # The last 105 layers represent the last resnet block and the layers after.
    for i in range(1,len(transfered_model.layers)-105):
        # print(transfered_model.layers[i], model.layers[8+i])
        # print(transfered_model.layers[i].output_shape, model.layers[8+i].output_shape)
        transfered_model.layers[i].set_weights(model.layers[8+i].weights)
        transfered_model.layers[i].trainable = False
    
    del model
    return transfered_model

def perform_transfered_task(dataset= "cifar10", epochs=100, batch_size=64):
    # Getting compile parameters and model with transfered weights
    optimizer, callbacks = get_compiler_parameters()
    model = transfer_weights(dataset= dataset)

    model.compile(optimizer= optimizer, metrics= ["accuracy"], loss= "categorical_crossentropy")
    model.summary()

    # Getting data
    x_train, y_train, x_validation, y_validation, unlabeled = get_semisupervised_data(dataset=dataset)

    x_test, y_test = get_test_data(dataset=dataset)
        
    # Training model with transfered weights on labeled data
    history = model.fit(x= x_train, y= y_train, epochs= epochs, batch_size= batch_size, validation_data= (x_validation, y_validation), callbacks= callbacks)
    
    model.save("models/CFN_transfered_" + dataset[:-2] + "_labeled.hdf5")

    if not(Path("models/history").exists()):
        mkdir("models/history")

    with open("models/history/history_CFN_transfered_" + dataset[:-2] + "_labeled.pkl", 'wb') as history_file:
        pkl.dump(history.history, history_file)

    scores = model.evaluate(x= x_test, y= y_test)
    
    if not(Path("models/scores").exists()):
        mkdir("models/scores")
    with open("models/scores/scores_CFN_transfered_" + dataset[:-2] + "_labeled.pkl", 'wb') as scores_file:
        pkl.dump(scores, scores_file)

    print("Test loss labeled: " + str(scores[0]))
    print("Test accuracy labeled: " + str(scores[1]))

    # Cancatenating unlabeled predicted labels to the train set
    x_train= concatenate((x_train, unlabeled), axis=0)
    y_unlabeled = model.predict(unlabeled)
    y_train= concatenate((y_train, y_unlabeled), axis=0)


    # Retraining model on the concatenated predicted labels with the initial labels
    history = model.fit(x= x_train, y= y_train, epochs= epochs, batch_size= batch_size, validation_data= (x_validation, y_validation), callbacks= callbacks)

    model.save("models/CFN_transfered_full.hdf5")

    with open("models/history/history_CFN_transfered_" + dataset[:-2] + "_full.pkl", 'wb') as history_file:
        pkl.dump(history.history, history_file)

    # Testing after concatenating the predicted labels
    scores = model.evaluate(x= x_test, y= y_test)

    with open("models/scores/scores_CFN_transfered_" + dataset[:-2] + "_full.pkl", 'wb') as scores_file:
        pkl.dump(scores, scores_file)
    
    print("Test loss full: " + str(scores[0]))
    print("Test accuracy full: " + str(scores[1]))
    
    return history, scores

perform_transfered_task(dataset="cifar10")