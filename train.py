#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from collections import namedtuple

import numpy
import theano
import theano.tensor as tensor
from theano.tensor.nnet import categorical_crossentropy

from core import EPS, DTYPE, IDX_TYPE, dtype_cast
from module import SAE, DAE



def sgd(params, cost, lr=0.01):
    return [(p, p - lr * g) for p, g in zip(params, tensor.grad(cost=cost, wrt=params))]


def adadelta(params, cost, lr=1.0, rho=0.95):
    # from https://github.com/fchollet/keras/blob/master/keras/optimizers.py
    grads = tensor.grad(cost, params)
    accus = [shared_zeros_like(p.get_value()) for p in params]
    delta_accus = [shared_zeros_like(p.get_value()) for p in params]
    updates = []
    for p, g, a, d_a in zip(params, grads, accus, delta_accus):
        new_a = rho * a + (1.0 - rho) * tensor.square(g)
        updates.append((a, new_a))
        update = g * tensor.sqrt(d_a + EPS) / tensor.sqrt(new_a + EPS)
        new_p = p - lr * update
        updates.append((p, new_p))
        new_d_a = rho * d_a + (1.0 - rho) * tensor.square(update)
        updates.append((d_a, new_d_a))
    return updates


def unzip(zipped):
    return [v[0] for v in zipped], [v[1] for v in zipped]


class Train(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        self.max_epochs = 0
        self.max_seq_length = 0
        self.batch_size = 2
        self.saveto = 0
        self.reloadfrom = 0
        self.save_freq = 0
        self.disp_freq = 0

    def to_one_hot(self, x, vocab_size):
        pass

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

    def prepare_data(self, dataset):
        pass

    def get_fupdate_rval(self, f_update, dataset, train_idx):
        pass

    def show_info(self, used_idx, epoc_idx, cost):
        pass

    def epoch(self, f_update, dataset, epoch_idx):
        itered_samples = 0
        used_idx = 0
        minibatches_idx = self.get_minibatch_idx(len(dataset), shuffle=False)

        for train_idx in minibatches_idx:
            _, cost = self.get_fupdate_rval(f_update, dataset, train_idx)
            itered_samples += len(train_idx)
            used_idx += 1
            self.show_info(used_idx, epoch_idx, cost)
        print('Seen %d samples' % itered_samples)

    def train_model(self, dataset):
        f_update = self.model.compile()
        start_time = time.time()
        try:
            for epoch_idx in range(self.max_epochs):
                self.epoch(f_update, dataset, epoch_idx)
        except KeyboardInterrupt:
            pass
        except StopIteration:
            pass
        end_time = time.time()


class TrainSAE(Train):
    def __init__(self, model, optimizer):
        Train.__init__(self, model, optimizer)

    def to_one_hot(self, x, vocab_size):
        """
        x: batch_size * max_sents_length
        return: max_sents_length * batch_size * vocab_size
        """
        batch_size, max_sents_length = x.shape
        onehot_x = numpy.zeros((batch_size, max_sents_length, vocab_size))
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
        return self.to_one_hot(x, self.model.vocab_size), mask

    def get_fupdate_rval(self, f_update, dataset, train_idx):
        x, mask = self.prepare_data([dataset[i] for i in train_idx])
        return f_update(x, mask)


class TrainDAE(Train):
    def __init__(self, model, optimizer):
        Train.__init__(self, model, optimizer)

    def to_one_hot(self, x, vocab_size):
        """
        x 1~vocab_size
        x: batch_size * max_doc_length * max_sents_length
        return: max_doc_length * batch_size * max_sents_length * vocab_size
        """
        batch_size, max_doc_length, max_sents_length = x.shape
        onehot_x = numpy.zeros((batch_size, max_doc_length, max_sents_length, vocab_size))
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
        return x, mask

    def prepare_data(self, dataset):
        length = [len(s) for s in dataset]
        n_samples = len(dataset)
        max_doc_length = max(length)
        max_sent_length = max([len(s[i])  for s in dataset for i in range(len(s))])
        print(max_sent_length)

        sents_list, st_mask_list = unzip([self._prepare_sentences_data(dataset[i], max_doc_length, max_sent_length)
                      for i in range(n_samples)])
        x = numpy.zeros((n_samples, max_doc_length, max_sent_length)).astype(IDX_TYPE)
        s_mask = numpy.zeros((n_samples, max_doc_length, max_sent_length)).astype(IDX_TYPE)
        d_mask = numpy.zeros((n_samples, max_doc_length)).astype(DTYPE)

        for i in range(n_samples):
            x[i] = sents_list[i]
            s_mask[i] = st_mask_list[i]
            d_mask[i, :length[i]] = 1
        return (self.to_one_hot(x, self.model.sae.vocab_size),
                s_mask.swapaxes(1, 0), d_mask.swapaxes(1, 0))

    def get_fupdate_rval(self, f_update, dataset, train_idx):
        x, s_mask, d_mask = self.prepare_data([dataset[i] for i in train_idx])
        return f_update(x, s_mask, d_mask)
