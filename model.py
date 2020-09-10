import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

kernel_shape = 3
pool_shape = 2

#Input and rotations
stamp_input = layers.Input(shape=(63,63,3), name="StampInput")
rot90 = layers.Lambda(lambda x: tf.image.rot90(x), name="Rotated90")(stamp_input)
rot180 = layers.Lambda(lambda x: tf.image.rot90(x), name="Rotated180")(rot90)
rot270 = layers.Lambda(lambda x: tf.image.rot90(x), name="Rotated270")(rot180)

#Input list
inputs = [stamp_input, rot90,rot180,rot270]
#layers in the model
layers_model = [
    layers.Conv2D(filters=32,kernel_size=4, activation="relu", name="Conv1"),
    layers.Conv2D(filters=32,kernel_size=kernel_shape,activation="relu",name="Conv2"),
    layers.MaxPool2D(pool_size=pool_shape, name="FirstPool"),
    layers.Conv2D(filters=64,kernel_size=kernel_shape,activation="relu",name="Conv3"),
    layers.Conv2D(filters=64,kernel_size=kernel_shape,activation="relu",name="Conv4"),
    layers.Conv2D(filters=64,kernel_size=kernel_shape,activation="relu",name="Conv5"),
    layers.MaxPool2D(pool_size=pool_shape, name="SecondPool"),
    layers.Flatten(name="Flatten"),
    layers.Dense(units=64, name="FirstDense"),
    layers.Dense(units=64, name="SecondDense")
]
#Storing output for each rotation
output_rot = []
for input_i in inputs:
    for l in layers_model:
        input_i = l(input_i)
    output_rot.append(input_i)

#Concatenate output from rotations
concat = tf.keras.backend.stack(output_rot,axis=1)
rot_pooling = layers.AveragePooling1D(pool_size=4, name="CyclicAveragePool")(concat)
flatten = layers.Flatten(name="FlattenPool")(rot_pooling)
probs = layers.Dense(units=5,activation="softmax",name="Logit")(flatten)

model = Model(inputs=stamp_input, outputs=probs)
