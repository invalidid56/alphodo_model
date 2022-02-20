import tensorflow as tf

def idcnn():
    model = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(
            filters=64,
            kernel_size=(3, 3)
        ),
        tf.keras.layers.MaxPool2D(
            strides=(2, 2)
        ),
        tf.keras.layers.SeparableConv2D(
            filters=64,
            kernel_size=(1, 1)
        ),
        tf.keras.layers.MaxPool2D(
            strides=(2, 2)
        )
    ]
    )
    # model.compile(optimizer=tf.keras.optimizers.Adam(), loss='cross-entropy')
    model.build((16, 16, 16, 16))
    model.summary()
    return model

idcnn()