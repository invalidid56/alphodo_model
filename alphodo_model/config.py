import json
import sys
import os


def main():
    data_dir = input('input data dir: ')
    ext_dir = input('input ext dir: ')
    model_dir = input('input mod dir: ')

    gpus = input('input n of gpus: ')
    batch_size = input('input batch size: ')
    train_step = input('input train step: ')
    eval_step = input('input eval step: ')

    config = {"DATA_HOME": data_dir,
              "EX_HOME": ext_dir,
              "MOD_HOME": model_dir,
              "GPU": gpus,
              "BATCH_SIZE": batch_size,
              "TRAIN_STEP": train_step,
              "EVAL_STEP": eval_step}

    with open(os.path.join(os.getcwd(), '../config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent='\t')

    return 0


def parse():
    with open(os.path.join(os.getcwd(), '../config.json'), 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        # TODO: Exception
    return json_data


if __name__ == '__main__':
    if sys.argv[1] == '0':
        main()
    else:
        parse()
