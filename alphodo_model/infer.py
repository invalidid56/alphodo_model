import keras
import numpy as np
import sys
import os
from PIL import Image
import base64

import config


def main():
    pass


def parse(image_dir):
    # input (248, 248, 3) ndarray
    image_pil = Image.open(image_dir)
    image_pil = image_pil.resize((248, 248))
    image = np.array(image_pil)

    image = image.reshape(-1, 248, 248, 3)

    load_model: keras.Model
    load_model = keras.models.load_model(
        os.path.join(os.path.join(config.parse()['MOD_HOME'], 'MODEL_IDCNN'), '1')
    )
    result = load_model.predict(image)
    print(result)
    return result


if __name__ == '__main__':
    parse(sys.argv[1])
