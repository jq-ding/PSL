from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, Dropout, add, Concatenate, UpSampling2D
from keras import regularizers


def preprocess_input(x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x


def bn_relu_conv(x, num_channels, kernel_size=3, strides=1, weight_decay=1e-4, block_name=None):
    x = BatchNormalization(name=block_name+'_bn')(x)
    x = Activation('relu', name=block_name+'_relu')(x)
    x = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(weight_decay), name=block_name+'_conv')(x)
    return x


def ResNet50():
    weight_decay = 1e-4
    num_channels_in = 16
    channel_multiplier = 2
    num_blocks = 4
    num_sub_blocks = 4

    num_classes = 12

    input_shape = (64, 64, 2)
    img_input = Input(shape=input_shape)

    x = Conv2D(num_channels_in, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), name='initial_conv')(img_input)

    for i in range(num_blocks):
        num_channels_out = channel_multiplier * num_channels_in
        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            name = 'block_{}_layer_{}'.format(i, j)
            y = bn_relu_conv(x, num_channels_in, kernel_size=1, strides=strides, weight_decay=weight_decay, block_name=name+'_down')
            y = bn_relu_conv(y, num_channels_in, kernel_size=3, strides=1, weight_decay=weight_decay, block_name=name)
            y = bn_relu_conv(y, num_channels_out, kernel_size=1, strides=1, weight_decay=weight_decay, block_name=name+'_up')
            if j == 0:
                x = Conv2D(num_channels_out, kernel_size=1, strides=strides, kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = add([x, y], name='block_{}_{}_add'.format(i, j))
            print(i, j)
        num_channels_in = num_channels_out

    # classification head
    x = BatchNormalization(name='final_bn')(x)
    x = Activation('relu', name='final_relu')(x)
    x = GlobalAveragePooling2D(name='final_global_pool')(x)
    #x = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay), name='prob')(x)
    x = Dense(num_classes, activation='softmax', name='prob')(x)

    model = Model(img_input, x, name='resnet50')

    return model


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    x = BatchNormalization(axis=3)(x)
    return x


def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x


def ResNet34():
    inpt = Input(shape=(64, 64, 2))
    # x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(inpt, nb_filter=32, kernel_size=(3, 3), strides=(2, 2), padding='same')  # (32,32,32)
    print("first:", x.shape)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # (56,56,64)
    x = Conv_Block(x, nb_filter=32, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=32, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=32, kernel_size=(3, 3))  # (32,32,32)
    fea1 = x
    print("fea1:", x.shape)

    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))  # (16,16,64)
    fea2 = x
    print("fea2:", x.shape)

    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))  # (8,8,128)
    fea3 = x
    print("fea3:", x.shape)

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))  # (4,4,256)
    fea4 = x
    print("fea4:", x.shape)
    # x = AveragePooling2D(pool_size=(7, 7))(x)
    gap1 = GlobalAveragePooling2D()(fea1)
    gap2 = GlobalAveragePooling2D()(fea2)
    gap3 = GlobalAveragePooling2D()(fea3)
    gap4 = GlobalAveragePooling2D()(fea4)
    print(gap4.shape)
    x = Concatenate()([gap1, gap2, gap3, gap4])
    print(x.shape)
    # x = Dense(1000, activation='softmax')(x)
    # x = Dense(256, activation='relu')(x)
    x = Dense(12, activation='softmax', name='prob')(x)

    recons = UpSampling2D(size=(2, 2))(fea4)
    recons = Concatenate()([recons, fea3])
    recons = Conv_Block(recons, nb_filter=128, kernel_size=(3, 3), with_conv_shortcut=True)
    recons = Conv_Block(recons, nb_filter=128, kernel_size=(3, 3))
    recons = Conv_Block(recons, nb_filter=128, kernel_size=(3, 3))
    recons = Conv_Block(recons, nb_filter=128, kernel_size=(3, 3))
    recons = Conv_Block(recons, nb_filter=128, kernel_size=(3, 3))
    recons = Conv_Block(recons, nb_filter=128, kernel_size=(3, 3))

    recons = UpSampling2D(size=(2, 2))(recons)
    recons = Concatenate()([recons, fea2])
    recons = Conv_Block(recons, nb_filter=64, kernel_size=(3, 3), with_conv_shortcut=True)
    recons = Conv_Block(recons, nb_filter=64, kernel_size=(3, 3))
    recons = Conv_Block(recons, nb_filter=64, kernel_size=(3, 3))
    recons = Conv_Block(recons, nb_filter=64, kernel_size=(3, 3))

    recons = UpSampling2D(size=(2, 2))(recons)
    recons = Concatenate()([recons, fea1])
    recons = Conv_Block(recons, nb_filter=32, kernel_size=(3, 3), with_conv_shortcut=True)
    recons = Conv_Block(recons, nb_filter=32, kernel_size=(3, 3))
    recons = Conv_Block(recons, nb_filter=32, kernel_size=(3, 3))

    recons = UpSampling2D(size=(2, 2))(recons)
    recons = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='prob5')(recons)



    # recons1 = UpSampling2D(size=(2, 2))(fea1)
    # recons1 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='prob1')(recons1)
    # print("recons1:", recons1.shape)
    #
    # recons2 = UpSampling2D(size=(2, 2))(fea2)
    # recons2 = Conv2D(32, (3, 3), activation='relu', padding='same')(recons2)
    # # recons2 = Conv2D(32, (3, 3), activation='relu', padding='same')(recons2)
    # # recons2 = Conv2D(32, (3, 3), activation='relu', padding='same')(recons2)
    # recons2 = UpSampling2D(size=(2, 2))(recons2)
    # recons2 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='prob2')(recons2)
    #
    # recons3 = UpSampling2D(size=(2, 2))(fea3)
    # recons3 = Conv2D(64, (3, 3), activation='relu', padding='same')(recons3)
    # # recons3 = Conv2D(64, (3, 3), activation='relu', padding='same')(recons3)
    # recons3 = UpSampling2D(size=(2, 2))(recons3)
    # recons3 = Conv2D(32, (3, 3), activation='relu', padding='same')(recons3)
    # # recons3 = Conv2D(32, (3, 3), activation='relu', padding='same')(recons3)
    # recons3 = UpSampling2D(size=(2, 2))(recons3)
    # recons3 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='prob3')(recons3)
    #
    # recons4 = UpSampling2D(size=(2, 2))(fea4)
    # recons4 = Conv2D(128, (3, 3), activation='relu', padding='same')(recons4)
    # # recons4 = Conv2D(128, (3, 3), activation='relu', padding='same')(recons4)
    # recons4 = UpSampling2D(size=(2, 2))(recons4)
    # recons4 = Conv2D(64, (3, 3), activation='relu', padding='same')(recons4)
    # # recons4 = Conv2D(64, (3, 3), activation='relu', padding='same')(recons4)
    # recons4 = UpSampling2D(size=(2, 2))(recons4)
    # recons4 = Conv2D(32, (3, 3), activation='relu', padding='same')(recons4)
    # # recons4 = Conv2D(32, (3, 3), activation='relu', padding='same')(recons4)
    # recons4 = UpSampling2D(size=(2, 2))(recons4)
    # recons4 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='prob4')(recons4)

    # model = Model(inputs=inpt, outputs=x)
    model = Model(inputs=inpt, outputs=[x, recons])

    return model