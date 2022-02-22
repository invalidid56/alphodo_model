import os.path

import datagen
import config
import tensorflow as tf
import model
import shutil


def main(initialize=False, version=1):
    conf = config.parse()
    data_dir = conf['DATA_HOME']
    ext_dir = conf['EX_HOME']
    model_dir = conf['MOD_HOME']

    train_ds, val_ds = datagen.main(data_dir, ext_dir, initialize=initialize)
    train_model = model.idcnn()
    epoch = int(conf['TRAIN_STEP'])

    train_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    history = train_model.fit(train_ds, epochs=epoch)

    test_loss, test_acc = train_model.evaluate(val_ds)
    print('\nTest accuracy: {}'.format(test_acc))

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    saved_model_path = os.path.join(model_dir, 'MODEL_IDCNN')

    export_path = os.path.join(saved_model_path, str(version))
    print('export_path = {}\n'.format(export_path))

    tf.keras.models.save_model(
        train_model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )


if __name__ == '__main__':
    main()
