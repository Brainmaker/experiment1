#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import json
from collections import defaultdict

import numpy
import theano
import theano.tensor as tensor

from core import EPS, DTYPE, IDX_TYPE, dtype_cast
from module import SAE, DAE


def _read_sent_data(file, vocabs):
    """
    vocabs: 一个字典，键为单词，值为序号
    """
    dataset = []
    for sent in file.readlines():
        dataset.append([vocabs[x] for x in sent.split()])
    return dataset


def _unzip(zipped):
    return [v[0] for v in zipped], [v[1] for v in zipped]


def _shared_zeros_like(x, name=None):
    return theano.shared(dtype_cast(numpy.zeros(x.shape)), name=name)


class Optimizer(object):
    def __init__(self):
        pass

    def optimize(self, params, cost):
        pass


class SGD(Optimizer):
    def __init__(self, lrate, momentum):
        Optimizer.__init__(self)
        self.lrate = lrate
        self.momentum = momentum

    def optimize(self, params, cost):
        return [(p, p - self.lrate * g) for p, g in zip(params, tensor.grad(cost=cost, wrt=params))]


class AdaDelta(Optimizer):
    # Not use
    def __init__(self, lrate, rho):
        Optimizer.__init__(self)
        self.lrate = lrate
        self.rho = rho

    def optimize(self, params, cost):
        grads = tensor.grad(cost, params)
        accus = [_shared_zeros_like(p.get_value()) for p in params]
        delta_accus = [_shared_zeros_like(p.get_value()) for p in params]
        updates = []
        for p, g, a, d_a in zip(params, grads, accus, delta_accus):
            new_a = self.rho * a + (1.0 - self.rho) * tensor.square(g)
            updates.append((a, new_a))
            update = g * tensor.sqrt(d_a + EPS) / tensor.sqrt(new_a + EPS)
            new_p = p - self.lrate * update
            updates.append((p, new_p))
            new_d_a = self.rho * d_a + (1.0 - self.rho) * tensor.square(update)
            updates.append((d_a, new_d_a))
        return updates


class Trainer(object):
    def __init__(self, model, max_epochs, batch_size, lrate, momentum,
                 dataset_path, vocabs_path, save_path, load_path):
        self.model = model
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lrate = lrate
        self.momentum = momentum
        self.dataset_path = dataset_path
        self.vocabs_path = vocabs_path
        self.save_path = save_path
        self.load_path = load_path

        self.optimizer = SGD(self.lrate, self.momentum)

        with open(vocabs_path, 'r') as f:
            self.vocabs = defaultdict(lambda : 0, json.loads(f.read()))

        if load_path:
            model.load(load_path)

        self.f_update = model.compile(self.optimizer.optimize)

    def to_one_hot(self, x, vocab_size):pass

    def prepare_data(self, dataset):pass

    def get_fupdate_rval(self, dataset, train_idx):pass

    def get_minibatch_idx(self, dataset_n_samples, shuffle=False):
        idx_list = numpy.arange(dataset_n_samples, dtype="int32")

        if shuffle:
            numpy.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(dataset_n_samples // self.batch_size):
            minibatches.append(idx_list[minibatch_start: minibatch_start + self.batch_size])
            minibatch_start += self.batch_size

        if minibatch_start != dataset_n_samples:
            minibatches.append(idx_list[minibatch_start:])

        return minibatches

    def run_epochs(self):
        for epoch_idx in range(self.max_epochs):
            print('In epoch', epoch_idx)

            for filename in os.listdir(self.dataset_path):
                data_file = open(self.dataset_path + '\\' + filename, 'r', encoding='utf-8')
                file_dataset = _read_sent_data(data_file, self.vocabs)
                print('Processing file:', filename)
                minibatches_idx = self.get_minibatch_idx(len(file_dataset), shuffle=False)

                for train_idx in minibatches_idx:
                    _, cost = self.get_fupdate_rval(file_dataset, train_idx)

                    if numpy.isnan(cost) or numpy.isinf(cost):
                        raise StopIteration

                self.model.save(self.save_path)

    def train(self):
        start_time = time.time()
        try:
            self.run_epochs()
        except KeyboardInterrupt:
            print('Training Interrupted by user')
        except StopIteration:
            print('Bad cost detected')
        finally:
            pass
        end_time = time.time()
        print('Training Time:', end_time - start_time)


class SAETrainer(Trainer):
    def __init__(self,
                 vocab_size, s_enc_dim, s_dec_dim, use_dropout,
                 max_epochs, batch_size, lrate, momentum,
                 dataset_path='',
                 vocabs_path='',
                 save_path='',
                 load_path=''):

        sae = SAE(vocab_size, s_enc_dim, s_dec_dim, use_dropout)

        Trainer.__init__(self,
                         model=sae,
                         max_epochs=max_epochs,
                         batch_size=batch_size,
                         lrate=lrate,
                         momentum=momentum,
                         dataset_path=dataset_path,
                         vocabs_path=vocabs_path,
                         save_path=save_path,
                         load_path=load_path)

    def to_one_hot(self, x, vocab_size):
        """
        x: batch_size * max_sents_length
        return: max_sents_length * batch_size * vocab_size
        """
        batch_size, max_sents_length = x.shape
        onehot_x = numpy.zeros((batch_size, max_sents_length, vocab_size), dtype=IDX_TYPE)
        for i in range(batch_size):
            for j in range(max_sents_length):
                onehot_x[i][j][x[i][j]-1] = 1
        return onehot_x.swapaxes(1, 0)

    def prepare_data(self, dataset):
        length = [len(s) for s in dataset]
        n_samples = len(dataset)
        maxlen = max(length)

        x = numpy.zeros((n_samples, maxlen)).astype(IDX_TYPE)
        mask = numpy.zeros((n_samples, maxlen)).astype(DTYPE)
        for idx, val in enumerate(dataset):
            x[idx, :length[idx]] = val
            mask[idx, :length[idx]] = 1
        return self.to_one_hot(x, self.model.vocab_size), mask.swapaxes(1, 0)

    def get_fupdate_rval(self, dataset, train_idx):
        x, mask = self.prepare_data([dataset[i] for i in train_idx])
        return self.f_update(x, mask)


class DAETrainer(Trainer):
    def __init__(self,
                 vocab_size, s_enc_dim, s_dec_dim, d_enc_dim, d_dec_dim, use_dropout,
                 max_epochs, batch_size, lrate, momentum,
                 dataset_path='',
                 vocabs_path='',
                 save_path='',
                 load_path='',
                 sae_load_path=''):

        self.sae = SAE(vocab_size, s_enc_dim, s_dec_dim, use_dropout)
        if sae_load_path:
            self.sae.load(sae_load_path)

        dae = DAE(d_enc_dim, d_dec_dim, self.sae, use_dropout)

        Trainer.__init__(self,
                         model=dae,
                         max_epochs=max_epochs,
                         batch_size=batch_size,
                         lrate=lrate,
                         momentum=momentum,
                         dataset_path=dataset_path,
                         vocabs_path=vocabs_path,
                         save_path=save_path,
                         load_path=load_path)

    def to_one_hot(self, x, vocab_size):
        """
        x 1~vocab_size
        x: batch_size * max_doc_length * max_sents_length
        return: max_doc_length * batch_size * max_sents_length * vocab_size
        """
        batch_size, max_doc_length, max_sents_length = x.shape
        onehot_x = numpy.zeros((batch_size, max_doc_length, max_sents_length, vocab_size), dtype=IDX_TYPE)
        for i in range(batch_size):
            for j in range(max_doc_length):
                for k in range(max_sents_length):
                    onehot_x[i][j][k][x[i][j][k]-1] = 1
        return onehot_x.swapaxes(1, 0)

    @staticmethod
    def _prepare_sentences_data(doc, max_nsents, max_sent_length):
        sents_length = [len(s) for s in doc]

        x = numpy.zeros((max_nsents, max_sent_length)).astype(IDX_TYPE)
        mask = numpy.zeros((max_nsents, max_sent_length)).astype(DTYPE)
        for idx, val in enumerate(doc):
            x[idx, :sents_length[idx]] = val
            mask[idx, :sents_length[idx]] = 1
        return x, mask.swapaxes(1, 0)

    def prepare_data(self, dataset):
        length = [len(s) for s in dataset]
        n_samples = len(dataset)
        max_doc_length = max(length)
        max_sent_length = max([len(s[i])  for s in dataset for i in range(len(s))])
        print(max_sent_length)

        sents_list, st_mask_list = _unzip([self._prepare_sentences_data(dataset[i], max_doc_length, max_sent_length)
                      for i in range(n_samples)])
        x = numpy.zeros((n_samples, max_doc_length, max_sent_length)).astype(IDX_TYPE)
        s_mask = numpy.zeros((n_samples, max_doc_length, max_sent_length)).astype(IDX_TYPE)
        d_mask = numpy.zeros((n_samples, max_doc_length)).astype(IDX_TYPE)

        for i in range(n_samples):
            x[i] = sents_list[i]
            s_mask[i] = st_mask_list[i]
            d_mask[i, :length[i]] = 1
        return (self.to_one_hot(x, self.model.sae.vocab_size),
                s_mask.swapaxes(1, 0), d_mask.swapaxes(1, 0))

    def get_fupdate_rval(self, dataset, train_idx):
        x, s_mask, d_mask = self.prepare_data([dataset[i] for i in train_idx])
        return self.f_update(x, s_mask, d_mask)
