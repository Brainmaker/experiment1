#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  ------ Train SAE and DAE ------
  Created by Xiaolin Wan, 5.10.2016

  1. 对句子数据封装的格式如下：
      dataset : 一个包含所有句子的二重列表。(一重)列表的每个元素也为一个列表，内容为：
                [词索引1，词索引2，.....，词索引m]，由此构成一个句子。
      dataset[n] : 指的是第n个句子。
      dataset[n][m] : 指的是第n个句子的第m个词，其值为词索引。

  2. 对段落数据的封装格式如下：
      dataset : 一个包含所有段落的三重列表。(一重)列表的每个元素为一个二重列表，
                表示该段落中的所有句子，封装方法同上，内容为：
                [句子1，句子2，.....，句子m]，由此构成一个段落。
      dataset[n] : 指的是第n个段落。
      dataset[n][m] : 指的是第n个段落的第m个句子。
      dataset[n][m][k] : 指的是第n个段落，第m个句子的第k个词，其值为词索引。

  3. 词表结构如下：
      word_dict : 一个包含所有可用词的字典。其键为单词，其值为编号。
      word_list : 一个包含所有可用词的列表。其索引为编号，其值为单词

  4. SAE接受的数据格式如下：
      target_seq : 一个包含minibatch中所有句子的三维ndarray数组，其三个维代表：
                   [batch_size * max_input_length * vocab_size]
      mask : minibatch中所有句子的mask矩阵，尺寸为：[batch_size * max_input_length]

  5. DAE接受的数据格式如下：
      target_seq : 一个包含minibatch中所有段落的四维ndarray数组，其四个维代表：
                   [batch_size * max_numof_sentence * max_sentence_length * vocab_size]
      mask : minibatch中所有句子的三维mask数组，尺寸为：[batch_size * max_doc_length * max_sent_length]

  Draft* Do NOT cite.
"""

import sys
import time

import numpy
import theano
import theano.tensor as tensor
from theano.tensor.nnet import categorical_crossentropy

from core import EPS, DTYPE, dtype_cast
from module import SAE, DAE

SENTENCES_TRAINING_SET = []
DOCUMENTS_TRAINING_SET = []

TRAINING_PARAMETERS = {
    # Parameters for SAE and DAE
    'vocab_size':120000,  # Vocabulary size
    'word_emb_dim':1000,  # Word embeding dimension and word-level LSTM number of hidden units.
    'sentence_emb_dim':1000,  # Sentence-level LSTM number of hidden units.
    'dec_nsteps':100,  # Max Output length of word decoder
    'use_dropout':True,

    # Parameters for training
    'optimizer':'sgd',
    'cost_function':'',
    'patience':20,
    'max_epochs':100,  # The maximum number of epoch to run
    'sgd_lr':0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    'decay':0,
    'moment':0.9,
    'valid_freq':370,  # Compute the validation error after this number of update.
    'batch_size':32,  # The batch size during training.
    'valid_batch_size':64,  # The batch size used for validation/test set.

    # Extra option
    'disp_freq':100,  # Display to stdout the training progress every N updates
    'save_freq':2000,  # Save the parameters after every saveFreq updates
    'reloadfrom':None,  # Path to a saved model we want to start from.
    'saveto':None,  # The best model will be saved there
}


def s_cost(target_seq, prob_pred_seq):
    """
    target_seq 大小为：max_timestep * batch_size * vocab_size
    """
    prob_pred_seq = tensor.clip(prob_pred_seq, EPS, 1.0 - EPS)
    cce = categorical_crossentropy(coding_dist=prob_pred_seq, true_dist=target_seq).mean(axis=1)
    return cce


def d_cost(target_seq, prob_pred_seq):
    """

    """
    prob_pred_seq = tensor.clip(prob_pred_seq, EPS, 1.0 - EPS)
    cce = categorical_crossentropy(coding_dist=prob_pred_seq, true_dist=target_seq).mean(axis=2).mean(axis=0)
    return cce


def get_minibatch_idx(batch_size):
    pass


def prepare_sent_data(seq):
    return seq, 0 # TODO


def prepare_doc_data(seq):
    return seq, 0 # TODO


def save_model():
    pass


def sgd(params, cost, lr, moment):
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


def sents_func_compile(sae, optimizer):
    input_sents = tensor.tensor3('input_sents', dtype=DTYPE)
    target_sents = tensor.tensor3('target_sents', dtype=DTYPE)
    mask = tensor.matrix('mask', dtype=DTYPE)

    options = 0 #TODO

    pred_sents, prob_pred_sents = sae.forward(input_sents, mask)
    cost = s_cost(target_sents, prob_pred_sents)

    f_s_updates = theano.function(
        name    = 'f_s_updates',
        inputs  = [input_sents, mask, target_sents],
        outputs = [pred_sents, cost],
        updates = optimizer(sae.get_params(), cost, options),
    )

    return f_s_updates


def docs_func_compile(dae, optimizer):
    input_docs = tensor.tensor4('input_docs', dtype=DTYPE)
    target_docs = tensor.tensor4('target_docs', dtype=DTYPE)
    sent_mask = tensor.tensor3('sent_mask', dtype=DTYPE)
    doc_mask = tensor.matrix('doc_mask', dtype=DTYPE)
    max_doc_length = tensor.iscalar('max_doc_length')

    options = 0 #TODO

    pred_docs, prob_pred_docs = dae.forward(input_docs, sent_mask, doc_mask, max_doc_length)
    cost = d_cost(target_docs, prob_pred_docs)

    f_d_updates = theano.function(
        name    = 'f_d_updates',
        inputs  = [input_docs, sent_mask, doc_mask, max_doc_length, target_docs],
        outputs = [pred_docs, cost],
        updates = optimizer(dae.get_params(), cost, options),
    )

    return f_d_updates


def s_epoch(minibatches_idx, f_s_updates, current_state):
    n_samples = 0

    used_idx = 0
    for train_idx in minibatches_idx:
        used_idx += 1

        target_seq, mask = prepare_sent_data([SENTENCES_TRAINING_SET[i] for i in train_idx])

        n_samples += target_seq[1]
        max_length = target_seq[0]  # not used

        pred_sents, cost = f_s_updates(target_seq, mask, target_seq)

        if numpy.isnan(cost) or numpy.isinf(cost):
            print('bad cost detected: ', cost)
            raise InterruptedError

        if numpy.mod(used_idx, TRAINING_PARAMETERS['disp_freq']) == 0:
            print('Epoch ', current_state, 'Update ', used_idx, 'Cost ', cost) #TODO

        if TRAINING_PARAMETERS['saveto'] and numpy.mod(used_idx, TRAINING_PARAMETERS['save_freq']) == 0:
            print('Saving...')
            save_model()

        print('Seen %d samples' % n_samples)


def d_epoch(minibatches_idx, current_state):
    n_samples = 0

    used_idx = 0
    for train_idx in minibatches_idx:
        used_idx += 1

        target_seq, mask = prepare_data([SENTENCES_TRAINING_SET[i] for i in train_idx])

        n_samples += target_seq[1]
        _ = target_seq[0]  # not used

        cost = 0

        if numpy.isnan(cost) or numpy.isinf(cost):
            print('bad cost detected: ', cost)
            raise InterruptedError

        if numpy.mod(used_idx, TRAINING_PARAMETERS['disp_freq']) == 0:
            print('Epoch ', current_state, 'Update ', used_idx, 'Cost ', cost) #TODO

        if TRAINING_PARAMETERS['saveto'] and numpy.mod(used_idx, TRAINING_PARAMETERS['save_freq']) == 0:
            print('Saving...')

        print('Seen %d samples' % n_samples)
