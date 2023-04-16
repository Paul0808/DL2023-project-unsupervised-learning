
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, BatchNormalization, Activation, Add, ZeroPadding2D, MaxPool2D, AveragePooling2D, Flatten, Dense, Rescaling
from keras import backend

from Code.color_affine_layer import RandomColorAffine

# Block that passes forward the residual input to the output
def identity_mapping(input, filters_no, channels_axis= 3):
    
    # First layer
    x = Conv2D(filters= filters_no, kernel_size= 3, padding= "same")(input)
    x = BatchNormalization(axis=channels_axis)(x)
    x = Activation("relu")(x)

    # Second layer
    x = Conv2D(filters= filters_no, kernel_size= 3, padding= "same")(x)
    x = BatchNormalization(axis=channels_axis)(x)

    # Adding residual input to output
    x = Add()([x, input])
    x = Activation("relu")(x)

    # Returning identity block
    return x

# Block that passes forward the residual input through convolutional layer and then to output
def convolutional_mapping(input, filters_no, channels_axis=3):
    # First layer
    x = Conv2D(filters=filters_no, kernel_size=3, strides=2, padding="same")(input)
    x = BatchNormalization(axis=channels_axis)(x)
    x = Activation("relu")(x)

    # Second layer
    x = Conv2D(filters=filters_no, kernel_size=3, padding="same")(x)
    x = BatchNormalization(axis=channels_axis)(x)

    # Passing residual input through convolutional layer and then adding to output
    input = Conv2D(filters=filters_no, kernel_size=1, strides=2)(input)
    x = Add()([x, input])
    x = Activation("relu")(x)

    return x

# Can return ResNet18 with architecture [2, 2, 2, 2] or ResNet34 with architecture [3, 4, 6, 3]
# By default returns ResNet34
def resnet_creator(input, input_shape=(7, 7, 3), architecture=[3, 4, 6, 3]):

    # Determining if the data has the color channels on axis 1 or 3
    if backend.image_data_format() == "channels_first":
        channels_axis = 1
    else:
        channels_axis = 3

    # Starting filters number
    filters_no = 64

    if architecture[0] == 2:
        name = "ResNet18"
    else:
        name = "ResNet34"

    # Declare input layer
    #input = Input(input_shape)

    # Normalize input to be between 0 and 1
    input = Rescaling(1/255)(input)
    
    # Weak jitter and brightness augmentation
    input = RandomColorAffine(brightness=0.2, jitter=0.1)(input)

    # Padd input
    x = ZeroPadding2D(3)(input)

    # First layer and Max Pool
    x = Conv2D(filters=filters_no, kernel_size=7, strides=2, padding="same")(input)
    x = BatchNormalization(axis=channels_axis)(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    # Adding the resnet blocks as dictated by the 'architecture'
    for _ in range(architecture[0]):
        # Adding identity block
        x = identity_mapping(input= x, filters_no= filters_no, channels_axis= channels_axis)

    for index in range(1, len(architecture)):
        filters_no *= 2
        # Adding convolutional block
        x = convolutional_mapping(input= x, filters_no= filters_no, channels_axis= channels_axis)
        for _ in range(architecture[index]-1):
            # Adding identity block
            x = identity_mapping(input= x, filters_no= filters_no, channels_axis= channels_axis)

    # Adding Average Pooling and dense layer
    x = AveragePooling2D(pool_size=2, padding= "same")(x)
    x = Flatten()(x)
    x = Dense(units= filters_no, activation="relu")(x)
    
    #resnet = Model(inputs= input, outputs= x, name=name)
    
    return x