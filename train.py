#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from collections import namedtuple

import numpy
import theano
import theano.tensor as tensor
from theano.tensor.nnet import categorical_crossentropy

from core import EPS, DTYPE, dtype_cast
from module import SAE, DAE

IDX_TYPE = 'int64'


def unzip(zipped):
    return [v[0] for v in zipped], [v[1] for v in zipped]


class Training(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        self.max_epochs = 0
        self.max_seq_length = 0
        self.batch_size = 0
        self.saveto = 0
        self.reloadfrom = 0
        self.save_freq = 0
        self.disp_freq = 0

    def get_minibatch_idx(self, dataset):
        return 0

    def prepare_data(self, dataset):
        return 0

    def get_fupdate_rval(self, f_update, dataset):
        return 0, 0


    def epoch(self, f_update, epoch_idx):
        n_samples = 0
        used_idx = 0
        minibatches_idx = self.get_minibatch_idx()

        pred_seq, cost = self.get_fupdate_rval(f_update)

        if numpy.isnan(cost) or numpy.isinf(cost):
            print('bad cost detected: ', cost)
            raise InterruptedError

        if numpy.mod(used_idx, self.disp_freq) == 0:
            print('Epoch ', epoch_idx, 'Update ', used_idx, 'Cost ', cost)  # TODO

        if self.saveto and numpy.mod(used_idx, self.save_freq) == 0:
            print('Saving...')
            self.model.save(self.saveto)

        print('Seen %d samples' % n_samples)


    def train_model(self):
        f_update = self.model.compile()
        start_time = time.time()
        try:
            for epoch_idx in range(self.max_epochs):
                self.epoch(f_update, epoch_idx)
        except KeyboardInterrupt:
            pass
        except StopIteration:
            pass
        end_time = time.time()


class SAETraining(Training):
    def __init__(self, model, optimizer):
        Training.__init__(self, model, optimizer)

    def get_minibatch_idx(self, dataset):
        pass

    def prepare_data(self, dataset):
        length = [len(s) for s in dataset]
        n_samples = len(dataset)
        maxlen = max(length)

        x = numpy.zeros((n_samples, maxlen)).astype(IDX_TYPE)
        mask = numpy.zeros((n_samples, maxlen)).astype(DTYPE)
        for idx, val in enumerate(dataset):
            x[idx, :length[idx]] = val
            mask[idx, :length[idx]] = 1
        return x, mask

    def get_fupdate_rval(self, f_update, dataset):
        minibatches_idx = self.get_minibatch_idx()
        target_seq, mask = self.prepare_data([dataset[i] for i in minibatches_idx])
        return f_update(target_seq, mask, target_seq)


class TrainDAE(Training):
    def __init__(self, model, optimizer):
        Training.__init__(self, model, optimizer)

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
        max_sent_length = 5 #TODO

        sents_list, st_mask_list = unzip([self._prepare_sentences_data(dataset[i], max_doc_length, max_sent_length)
                      for i in range(n_samples)])
        x = numpy.zeros((n_samples, max_doc_length, max_sent_length)).astype(IDX_TYPE)
        s_mask = numpy.zeros((n_samples, max_doc_length, max_sent_length)).astype(IDX_TYPE)
        d_mask = numpy.zeros((n_samples, max_doc_length)).astype(DTYPE)

        for i in range(n_samples):
            x[i] = sents_list[i]
            s_mask[i] = st_mask_list[i]
            d_mask[i, :length[i]] = 1
        return x, s_mask, d_mask
