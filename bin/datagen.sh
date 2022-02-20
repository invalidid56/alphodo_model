#!/usr/bin/env bash

cd ..
BASEDIR=$(pwd)

EX_HOME=$(cat config.json | jq .EX_HOME)  # 정제
EX_HOME=${EX_HOME:1:-1}

DATA_DIR=$(cat config.json | jq .DATA_HOME) # 원본
DATA_DIR=${DATA_DIR:1:-1}

mkdir -p $DATA_DIR $TMP_DIR

FILECNT=$(ls $TMP_DIR | wc -l)

if [ $FILECNT = 0 ] ; then
    echo '>>>> Start Datagen for Training.'

    python datagen.py \
      --data_dir=$DATA_DIR \
      --extract_dir=$EX_HOME

    echo '>>>> End Datagen for Training.'
else
    echo '>>>> Dataset files are already exist in target dir. Check and try datagen again.'
fi

# 원본 데이터(이미지) (DATA_DIR) -> 정규화된 nparray pickle (EX_dir)