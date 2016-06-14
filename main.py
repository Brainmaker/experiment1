#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
import getopt

from train import SGD, AdaDelta, SAETrainer, DAETrainer

SHORT_ARGS = 'ht:m:'
LONG_ARGS = ['help', 'test=', 'model=']

sgd = SGD(lrate=0.35)
adadelta = AdaDelta(lrate=0.1 ,rho=0.95)


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
        emb_dim=5,
        s_enc_dim=3,
        s_dec_dim=3,
        use_dropout=True,
        max_epochs=20,
        batch_size=4,
        optimizer=sgd,
        dataset_path=current_path + '/' + 'sae_test_sample',
        vocabs_path=current_path + '/' + 'test_vocabs.json',
        word_table_path=current_path + '/' + 'test_word_table.json',
        save_path=current_path + '/' + 'test_sae.json',
        load_path=current_path + '/' + 'test_sae.json',
        log_path=current_path + '/' + 'log.txt'
    )
    sae_trainer.train()


def test_dae():
    current_path = os.getcwd()
    dae_trainer = DAETrainer(
        emb_dim=5,
        s_enc_dim=3,
        s_dec_dim=3,
        d_enc_dim=3,
        d_dec_dim=3,
        use_dropout=True,
        max_epochs=5,
        batch_size=4,
        optimizer='',
        dataset_path=current_path + '/' + 'dae_test_sample',
        vocabs_path=current_path + '/' + 'test_vocabs.json',
        word_table_path=current_path + '/' + 'test_word_table.json',
        save_path=current_path + '/' + 'test_dae.json',
        load_path=current_path + '/' + 'test_dae.json',
        sae_load_path=current_path + '/' + 'test_sae.json',
        log_path=current_path + '/' + 'log.txt'
    )
    dae_trainer.train()


def train_sae():
    current_path = os.getcwd()
    sae_trainer = SAETrainer(
        emb_dim=50,
        s_enc_dim=1000,
        s_dec_dim=1000,
        use_dropout=True,
        max_epochs=10,
        batch_size=128,
        optimizer=sgd,
        dataset_path=current_path + '/' + 'sentence_dataset',
        vocabs_path=current_path + '/' + 'vocabs.json',
        word_table_path=current_path + '/' + 'word_table.json',
        save_path=current_path + '/' + 'trained_sae.json',
        load_path=current_path + '/' + 'trained_sae.json',
        log_path=current_path + '/' + 'log.txt'
    )
    sae_trainer.train()


def train_dae():
    current_path = os.getcwd()
    dae_trainer = DAETrainer(
        emb_dim=50,
        s_enc_dim=1000,
        s_dec_dim=1000,
        d_enc_dim=1000,
        d_dec_dim=1000,
        use_dropout=True,
        max_epochs=10,
        batch_size=32,
        optimizer=sgd,
        dataset_path=current_path + '/' + 'document_dataset',
        vocabs_path=current_path + '/' + 'vocabs.json',
        word_table_path=current_path + '/' + 'word_table.json',
        save_path=current_path + '/' + 'trained_dae.json',
        load_path=current_path + '/' + 'trained_dae.json',
        sae_load_path=current_path + '/' + 'trained_sae.json',
        log_path=current_path + '/' + 'log.txt'
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
