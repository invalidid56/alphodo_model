import sys
import datagen
import config
from keras import optimizers
import tensorflow as tf
import model


def main(initialize=False):
    conf = config.parse()
    data_dir = conf['DATA_HOME']
    ext_dir = conf['EX_HOME']
    train_ds, val_ds = datagen.main(data_dir, ext_dir, initialize=initialize)
    train_model = model.idcnn()
    epoch = int(conf['TRAIN_STEP'])
    lr = 0.01

    train_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    history = train_model.fit(train_ds, epochs=epoch)


if __name__ == '__main__':
    main()
