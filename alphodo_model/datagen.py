import os
import urllib.error

import wget
import time
import re
import config

import tensorflow as tf
import shutil

from keras.preprocessing.image import ImageDataGenerator


def main(data_home, ex_home, batch_size=128, initialize=False):
    # Download Data
    # 계속 보내면 423으로 막아버림, 중간중간 끊어서 다운로드 필요
    if initialize:
        if os.path.isdir(data_home):
            shutil.rmtree(data_home)
        os.mkdir(data_home)

        with open('../img_urls.txt') as f:
            for i, url in enumerate(f.readlines()):
                print(i)
                try:
                    wget.download(url, data_home)
                    time.sleep(1.5)
                except urllib.error.HTTPError:
                    #time.sleep(600) # 미봉책
                    #wget.download(url, data_home)
                    print('WGET DOWNLOAD ERROR')
                    continue
                if i > 100:
                    break

    # Labeling Images

    reg = {
        'ulcer': [re.compile('IMG2007_')],
        'ciner': [re.compile('IMG2005_')],
        'powder': [re.compile('IMG2006_')],
        'mosaic': [re.compile('IMG18'), re.compile('DSC18_07[69]')],
        'rancid': [re.compile('IMG19')],
        'mildew': [re.compile('DSC18_07[78]'), re.compile('DSC18_08'), re.compile('HLDSC')],
        'fungi': [re.compile('IMG_')]
    }
    diseases = reg.keys()

    file_list = [file for file in os.listdir(data_home) if (file.endswith('.jpg') or file.endswith('.jpeg'))]
    if os.path.isdir(ex_home):
        shutil.rmtree(ex_home)
    os.mkdir(ex_home)

    for disease in diseases:
        os.mkdir(os.path.join(ex_home, disease))

    for img in file_list:

        for d in diseases:
            for r in reg[d]:
                if r.match(img):
                    old_path = os.path.join(data_home, img)
                    new_path = os.path.join(ex_home, d)
                    shutil.copy(
                            old_path, os.path.join(new_path, img)
                    )
                    break

    # Build Dataset

    img_height = 248
    img_width = 248
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        ex_home,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        ex_home,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Data Aug
    '''
    image_generator = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    train_result = image_generator.flow_from_directory(
        ex_home,
        target_size=(248, 248),
        batch_size=batch_size,
        # binary_crossentropy 손실 함수를 사용하므로 binary 형태로 라벨을 불러와야 합니다.
    )

    train_ds = ImageDataGenerator(
        rescale=1. / 255.,
        width_shift_range=0.3,
        zoom_range=0.7,
        horizontal_flip=True
    )
    val_ds = ImageDataGenerator(
        rescale=1. / 255.,
    )
    train_result = train_ds.flow_from_directory(
        ex_home,
        target_size=(248, 248),
        batch_size=batch_size,
    )
    test_result = val_ds.flow_from_directory(
        ex_home,
        target_size=(248, 248),
        batch_size=batch_size,
    )
    '''
    # Save
    norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (norm_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (norm_layer(x), y))

    return train_ds, val_ds


if __name__ == '__main__':
    main(config.parse()['DATA_HOME'], config.parse()['EX_HOME'], batch_size=int(config.parse()['BATCH_SIZE']), initialize=False)

# 논문 페이지에서 이미지 다운로드 받고(DATA_HOME), nparray로 저장

