
from keras.models import Model
from keras.layers import Input, Concatenate, Dense, Dropout
from keras import backend

from resnet_backbone import resnet_creator

class CFN(Model):
    def __init__(self, *args, permutations_no=100, crop_dimensions=7, crops_no=9, channels_no=3, siamese_architecture=[3, 4, 6, 3], **kwargs):
        # Declare the input for the 9 siamese blocks
        inputs= [Input((crop_dimensions, crop_dimensions, channels_no)) for _ in range(crops_no)]
        
        # Generating the resnet block
        resnet = resnet_creator(input_shape= (crop_dimensions, crop_dimensions, channels_no), architecture= siamese_architecture)
        siammese_blocks = [resnet(block_input) for block_input in inputs]
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

        super().__init__(*args, inputs=inputs, outputs=x, name="CFN", **kwargs)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
    
# TODO: delete input in resnet block and use it as simple layers without creating a model, to have a nicer name for the block
# use keras.backend.name_scope
model = CFN()
print(model.layers[1])
model.summary()