import keras
import numpy

import config


def main():
    pass


def parse(input_image: numpy.ndarray):
    # input (248, 248, 3) ndarray
    load_model: keras.Model
    load_model = keras.models.load_model(
        config.parse()['MOD_HOME']
    )

    return load_model.predict(input_image)


if __name__ == '__main__':
    main()
