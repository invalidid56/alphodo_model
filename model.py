import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D, \
    AveragePooling2D, Flatten, SeparableConv2D, BatchNormalization


def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None,
                     kernel_init='glorot_uniform', bias_init='zeros'):

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3_reduce)

    conv_5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5_reduce)

    max_pool = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)

    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(max_pool)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output


def idcnn():
    input_layer = Input(shape=(248, 248, 3))    # BATCH?

    # Layer1
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='SepCon1')(input_layer)
    x = MaxPool2D((3, 3), strides=(2, 2), name='MaxPool1')(x)

    # Layer2
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', name='SepCon2')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), name='MaxPool2')(x)
    x = BatchNormalization()(x)

    # Layer3
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_1')
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_2')
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_3')
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_4')
    x = MaxPool2D((3, 3), strides=(2, 2), name='MaxPool3')(x)

    # Layer4
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_5')
    x = MaxPool2D((3, 3), strides=(2, 2), name='MaxPool4')(x)

    # Layer5
    x = GlobalAveragePooling2D(name='GAP')(x)
    x = Dropout(0.4)(x)
    x = Dense(7, activation='softmax', name='output')(x)

    model = Model(input_layer, x, name='model')
    model.summary()

    return model

idcnn()
