#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import getopt

from train import SAETrainer, DAETrainer

SHORT_ARGS = 'ht:m:'
LONG_ARGS = ['help', 'test=', 'model=']


def usage():
    print("""
Usage:
  main [options]

Options:
  -h, --help                             Show help
  -t <model name>, --test=<model name>   Run test sample
  -m <model name>, --model=<model name>  Train sae or dae model
    """)


def test_sae():
    current_path = os.getcwd()
    sae_trainer = SAETrainer(
        vocab_size='',
        s_enc_dim='',
        s_dec_dim='',
        use_dropout='',
        max_epochs='',
        batch_size='',
        lrate='',
        momentum='',
        dataset_path=current_path + '\\' + '',
        vocabs_path=current_path + '\\' + '',
        save_path=current_path + '\\' + '',
        load_path=current_path + '\\' + ''
    )
    sae_trainer.train()


def test_dae():
    current_path = os.getcwd()
    dae_trainer = DAETrainer(
        vocab_size='',
        s_enc_dim='',
        s_dec_dim='',
        d_enc_dim='',
        d_dec_dim='',
        use_dropout='',
        max_epochs='',
        batch_size='',
        lrate='',
        momentum='',
        dataset_path=current_path + '\\' + '',
        vocabs_path=current_path + '\\' + '',
        save_path=current_path + '\\' + '',
        load_path=current_path + '\\' + '',
        sae_load_path=current_path + '\\' + ''
    )
    dae_trainer.train()


def train_sae():
    current_path = os.getcwd()
    sae_trainer = SAETrainer(
        vocab_size='',
        s_enc_dim='',
        s_dec_dim='',
        use_dropout='',
        max_epochs='',
        batch_size='',
        lrate='',
        momentum='',
        dataset_path=current_path + '\\' + '',
        vocabs_path=current_path + '\\' + '',
        save_path=current_path + '\\' + '',
        load_path=current_path + '\\' + ''
    )
    sae_trainer.train()


def train_dae():
    current_path = os.getcwd()
    dae_trainer = DAETrainer(
        vocab_size='',
        s_enc_dim='',
        s_dec_dim='',
        d_enc_dim='',
        d_dec_dim='',
        use_dropout='',
        max_epochs='',
        batch_size='',
        lrate='',
        momentum='',
        dataset_path=current_path + '\\' + '',
        vocabs_path=current_path + '\\' + '',
        save_path=current_path + '\\' + '',
        load_path=current_path + '\\' + '',
        sae_load_path=current_path + '\\' + ''
    )
    dae_trainer.train()


def main(argv):
    if not argv:
        usage()
        sys.exit()

    try:
        opts, args = getopt.getopt(argv, SHORT_ARGS, LONG_ARGS)
    except getopt.GetoptError:
        print('ERROR: unkown option or wrong option format \"%s\"' % str(argv[0]))
        sys.exit(2)

    for opt, arg in opts:
        print(arg)
        if opt in ('-h', '--help'):
            usage()
            break

        elif opt in ('-t', '--test'):
            if arg == 'sae':
                test_sae()
            elif arg == 'dae':
                test_dae()
            else:
                print('ERROR: unkonw model name')
            break

        elif opt in ('-m', '--model'):
            if arg == 'sae':
                train_sae()
            elif arg == 'dae':
                train_dae()
            else:
                print('ERROR: unkonw model name')
            break


if __name__ == '__main__':
    main(sys.argv[1:])
