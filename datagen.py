import os
import urllib.error

import wget
import time
import re
import config

import PIL
import PIL.Image
import tensorflow as tf


def main(data_home, initialize=False):
    # Download Data
    # 계속 보내면 423으로 막아버림, 중간중간 끊어서 다운로드 필요

    if initialize:
        if os.path.isdir(data_home):
            os.rmdir(data_home)
        os.mkdir(data_home)

        with open('img_urls.txt') as f:
            for i, url in enumerate(f.readlines()):
                print(i)
                try:
                    wget.download(url, data_home)
                    time.sleep(0.8)
                except urllib.error.HTTPError:
                    #time.sleep(600) # 미봉책
                    #wget.download(url, data_home)
                    print('error occured')
                    continue
                if i > 100:
                    break
    # Data Aug

    # Labeling Images

    reg = {
        'ulcer': [re.compile('2007_')],
        'ciner': [re.compile('2005_')],
        'powder': [re.compile('2006_')],
        'mosaic': [re.compile('IMG18'), re.compile('DSC18')],
        'rancid': [re.compile('IMG19')],
        'mildew': [re.compile('DSC18_07'), re.compile('DSC18_08'), re.compile('HLDSC')],
        'fungi': [re.compile('IMG_')]
    }
    diseases = reg.keys()

    file_list = [file for file in os.listdir(data_home) if file.endswith('.jpg')]
    for disease in diseases:
        os.mkdir(os.path.join(data_home, disease))

    for img in file_list:
        for d in diseases:
            for r in reg[d]:
                if r.match(img):
                    new_path = os.path.join(data_home, reg[d])
                    os.renames(
                         os.path.join(data_home, img), os.path.join(new_path, img)
                    )
                    break

    # Build Dataset

    img_height = 248
    img_width = 248
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_home,
        validation_split = 0.2,
        subset = 'training',
        seed = 111,
        image_size = (img_height, img_width)
    )

    # Convering Img+Lbl 2 nparray


if __name__ == '__main__':
    main(config.parse()['DATA_HOME'], initialize=True)

# 논문 페이지에서 이미지 다운로드 받고(DATA_HOME), nparray로 저장

