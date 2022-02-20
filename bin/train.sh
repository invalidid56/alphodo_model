#!/usr/bin/env bash

cd ..
BASEDIR=$(pwd)

EX_HOME=$(cat config.json | jq .EX_HOME)  # 정제
EX_HOME=${EX_HOME:1:-1}

MOD_DIR=$(cat config.json | jq .MOD_HOME) # 원본
MOD_DIR=${MOD_DIR:1:-1}

mkdir -p $DATA_DIR $TMP_DIR

FILECNT=$(ls $TMP_DIR | wc -l)

if [ $FILECNT = 0 ] ; then
    echo '>>>> Start Datagen for Training.'

    python train.py \
      --extract_dir=$EX_HOME  \
      --model_dir=$MOD_HOME

    echo '>>>> End Datagen for Training.'
else
    echo '>>>> Dataset files are already exist in target dir. Check and try datagen again.'
fi

# datagen -> train