from tensorflow import keras
from tensorflow.keras import optimizers
from keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import ModelCheckpoint
from psl.dataset import load_data
from psl.utils import preprocess_input
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, Dropout, add, Concatenate, UpSampling2D, Reshape, multiply
import numpy as np


import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import keras

# config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess =tf.compat.v1.Session(config=config)


# 1. set up data
x_train, y_train = load_data('train')
x_train = preprocess_input(x_train)

x_val, y_val = load_data("val")
x_val = preprocess_input(x_val)

x_train = np.concatenate((x_train, x_val))
y_train = np.concatenate((y_train, y_val))
print("x_train:", x_train.shape)

index = [i for i in range(x_train.shape[0])]
np.random.shuffle(index)
x_train = x_train[index]
y_train = y_train[index]

num_classes = 12
y_train = keras.utils.to_categorical(y_train, num_classes)
print("final shape:", x_train.shape)


def preprocess_input(x):
    x = x.astype(np.float32)
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def squeeze_excite_block(input, ratio=4):
    init = input
    filters = init._keras_shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    xx = multiply([init, se])
    return xx


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filters=nb_filter, kernel_size=kernel_size, padding=padding, strides=strides, activation='relu')(x)
    x = BatchNormalization(axis=3)(x)
    return x


def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    x = squeeze_excite_block(x, ratio=4)
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def ResNet34():
    inpt = Input(shape=(64, 64, 2))

    x = Conv2d_BN(inpt, nb_filter=32, kernel_size=(3, 3), padding='same')
    print("first:", x.shape)

    x = Conv_Block(x, nb_filter=32, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=32, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=32, kernel_size=(3, 3))  # (32,32,32)
    fea1 = x
    print("fea1:", x.shape)

    # x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))  # (16,16,64)
    fea2 = x
    print("fea2:", x.shape)

    # x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))  # (8,8,128)
    fea3 = x
    print("fea3:", x.shape)

    # x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))  # (4,4,256)
    fea4 = x
    print("fea4:", x.shape)

    gap1 = GlobalAveragePooling2D()(fea1)
    gap2 = GlobalAveragePooling2D()(fea2)
    gap3 = GlobalAveragePooling2D()(fea3)
    gap4 = GlobalAveragePooling2D()(fea4)

    x = Concatenate(name='gap')([gap1, gap2, gap3, gap4])
    x = Dense(12, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    model.summary()
    return model


# 2. set up model
model = ResNet34()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

metric = 'val_accuracy'
filepath="/home/dingjiaqi/Program/deepyeast-master/deepyeast-master/deepyeast/weights/A-{accuracy:.3f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor=metric, verbose=1, save_best_only=True, mode='max')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, cooldown=0, min_lr=1e-5)
callbacks_list = [checkpoint, reduce_lr]


# 3. training loop
batch_size = 32
epochs = 50
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          validation_split=0.1,
          # validation_data=(x_val, y_val)
          callbacks=callbacks_list)
