#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  ------ Document Autoencoder ------
  Created by Xiaolin Wan, 5.1.2016

  Build a deeply document autoencoder with Bi-LSTM.
  We adopt a Bi-LSTM structure with four layer for encoding and four layer for decoding,
  a deeply structure is planted between encoder and decoder to extract high-level sentence representation.

  Draft* Do NOT cite.
"""

import sys
import time

import numpy
import theano
import theano.tensor as tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
epsilon = 1e-6
dtype = theano.config.floatX

def np2shared(value, name=None):
    return theano.shared(value.astype(dtype), name=name)


def shared_zeros(shape, name=None):
    return np2shared(value=numpy.zeros(shape), name=name)


def shared_zeros_like(x, name=None):
    return shared_zeros(shape=x.shape, name=name)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_ortho_weight(dim):
    """
    Get a (n * 4n) matrix
    """
    tmp = numpy.random.randn(dim, dim)
    u, s, v = numpy.linalg.svd(tmp)
    u = u.astype(config.floatX)
    concat_ortho_weight = numpy.concatenate([u, u, u, u], axis=1)
    return concat_ortho_weight


def tensor_slice(_x, slice_tag, dim):
    """
    使用注意！例如W的大小是(N * 4N)，则dim=N
    """
    if _x.ndim == 3:
        return _x[:, :, slice_tag*dim : (slice_tag+1)*dim]
    return _x[:, slice_tag*dim : (slice_tag+1)*dim]


def adadelta(params, cost, lr=1.0, rho=0.95):
    """
    from https://github.com/fchollet/keras/blob/master/keras/optimizers.py
    """
    grads = tensor.grad(cost, params)
    accus = [shared_zeros_like(p.get_value()) for p in params]
    delta_accus = [shared_zeros_like(p.get_value()) for p in params]
    updates = []
    for p, g, a, d_a in zip(params, grads, accus, delta_accus):
        new_a = rho * a + (1.0 - rho) * tensor.square(g)
        updates.append((a, new_a))
        update = g * tensor.sqrt(d_a + epsilon) / tensor.sqrt(new_a + epsilon)
        new_p = p - lr * update
        updates.append((p, new_p))
        new_d_a = rho * d_a + (1.0 - rho) * tensor.square(update)
        updates.append((d_a, new_d_a))
    return updates


def categorical_crossentropy(y_true, y_pred):
    # from https://github.com/fchollet/keras/blob/master/keras/objectives.py
    y_pred = tensor.clip(y_pred, epsilon, 1.0 - epsilon)

    cce = tensor.nnet.categorical_crossentropy(y_pred, y_true)
    return tensor.mean(cce)


def mean_square_error(y_true, y_pred):
    return tensor.mean(tensor.square(y_pred - y_true))

class LSTMEncoder:
    def __init__(self, dim):
        """
        不考虑embedding
        """
        self.dim = dim
        self.nn_weights = {}.fromkeys(['W', 'U', 'b'])

        self.nn_weights['W'] = get_ortho_weight(dim)
        self.nn_weights['U'] = get_ortho_weight(dim)
        self.nn_weights['b'] = numpy.zeros((4 * dim,))


    def encode_step(self, x, m_, h_tm1, c_tm1):
        """
        Basic Equation:

            h(t), c(t) = LSTM(x(t), h(t-1), c(t-1))

        """
        preact = tensor.dot(h_tm1, self.nn_weights['U'])
        preact += x

        i = tensor.nnet.sigmoid(tensor_slice(preact, 0, self.dim))
        f = tensor.nnet.sigmoid(tensor_slice(preact, 1, self.dim))
        o = tensor.nnet.sigmoid(tensor_slice(preact, 2, self.dim))
        c = tensor.tanh(tensor_slice(preact, 3, self.dim))

        c = f * c_tm1 + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_tm1

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_tm1

        return h, c


    def encode(self, x, x_mask):
        nsteps = x.shape[0]
        if x.ndim == 3:
            n_samples = x.shape[1]
        else:
            n_samples = 1

        init_val = [tensor.alloc(numpy_floatX(0.), n_samples, self.dim),
                    tensor.alloc(numpy_floatX(0.), n_samples, self.dim)]

        state = (tensor.dot(x, self.nn_weights['W']) +
                 self.nn_weights['b'])

        layer_output, update = theano.scan(name          = 'LSTMEncoder',
                                           fn            = self.encode_step,
                                           sequences     = [state, x_mask],
                                           outputs_info  = init_val,
                                           non_sequences = None,
                                           n_steps       = nsteps)
        return layer_output


class LSTMDecoder:
    def __init__(self, dim):
        """
        不考虑embedding
        """
        self.dim = dim
        self.nn_weights = {}.fromkeys(['W', 'U', 'C', 'b'])

        self.nn_weights['W'] = get_ortho_weight(dim)
        self.nn_weights['U'] = get_ortho_weight(dim)
        self.nn_weights['C'] = get_ortho_weight(dim)
        self.nn_weights['b'] = numpy.zeros((4 * dim,))


    def _decode_step(self, y_tm1, m_, h_tm1, c_tm1, state):
        """
        Basic Equation:

            h(t), c(t) = Dec( LSTM(y(t), h(t-1), c(t-1)) )

        """
        preact = (tensor.dot(y_tm1, self.nn_weights['W']) +
                  tensor.dot(h_tm1, self.nn_weights['U']))
        preact += state

        i = tensor.nnet.sigmoid(tensor_slice(preact, 0, self.dim))
        f = tensor.nnet.sigmoid(tensor_slice(preact, 1, self.dim))
        o = tensor.nnet.sigmoid(tensor_slice(preact, 2, self.dim))
        c = tensor.tanh(tensor_slice(preact, 3, self.dim))

        c = f * c_tm1 + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_tm1

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_tm1

        return h, c


    def decode(self, y_tm1, h_tm1, c_tm1, mask, context_vector):
        """
        特点概述：对每个mask的句子走一步看一步，直到输出终止符
        """
        h, c = self._decode_step(y_tm1, h_tm1, c_tm1, mask, context_vector)
        return h, c


class MLP:
    def __init__(self, params):
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.mlp_n_layers = params['mlp_n_layers']

        self.nn_weights = {}.fromkeys(['w1','w2','w3','b1','b2','b3'])
        self.nn_weights['w1'] = self._init_mlp_weights(self.input_dim, self.hidden_dim)


    def _init_mlp_weights(self, dim1, dim2):
        bound = numpy.sqrt(1.0/dim1)
        w = numpy.random.uniform(-bound, bound, dim1)
        return w


    def run_mlp(self, x):
        h1 = tensor.nnet.relu(tensor.dot(self.nn_weights['w1'], x) + self.nn_weights['b1'])
        h2 = tensor.nnet.relu(tensor.dot(self.nn_weights['w2'], h1) + self.nn_weights['b2'])
        h3 = tensor.nnet.relu(tensor.dot(self.nn_weights['w3'], h2) + self.nn_weights['b3'])
        return h3


class Seq2Seq:
    def __init__(self, params):
        self.emb_dim = params['emb_dim']
        self.enc_dim = params['enc_dim']
        self.dec_dim = params['dec_dim']
        self.mlp_dim = params['mlp_dim']

        self.emb_matrix = tensor.matrix('emb_matrix')

        self.enc_layer_1 = LSTMEncoder(self.enc_dim)
        self.enc_layer_2 = LSTMEncoder(self.enc_dim)
        self.enc_layer_3 = LSTMEncoder(self.enc_dim)
        self.enc_layer_4 = LSTMEncoder(self.enc_dim)

        self.mlp = MLP(params)

        self.dec_layer_1 = LSTMDecoder(self.dec_dim)
        self.dec_layer_2 = LSTMDecoder(self.dec_dim)
        self.dec_layer_3 = LSTMDecoder(self.dec_dim)
        self.dec_layer_4 = LSTMDecoder(self.dec_dim)


    def decoder_step(self, y_tm1,
                     c1_tm1, c2_tm1, c3_tm1, c4_tm1,
                     h1_tm1, h2_tm1, h3_tm1, h4_tm1,
                     context_vector):
        mask = 0
        h1, c1 = self.dec_layer_1.decode(y_tm1, h1_tm1, c1_tm1, mask, context_vector)
        h2, c2 = self.dec_layer_1.decode(y_tm1, h2_tm1, c2_tm1, mask, context_vector)
        h3, c3 = self.dec_layer_1.decode(y_tm1, h3_tm1, c3_tm1, mask, context_vector)
        h4, c4 = self.dec_layer_1.decode(y_tm1, h4_tm1, c4_tm1, mask, context_vector)

        y = h1 # convert to y
        return y, c1, c2, c3, c4, h1, h2, h3, h4


    def build_model(self):
        x = tensor.matrix('x', dtype='')
        x_mask = tensor.matrix('x_mask', dtype='')
        y = tensor.matrix('y', dtype='')
        y_mask = tensor.matrix('y_mask', dtype='')

        emb_x = tensor.dot(x, self.emb_matrix)

        enc_h1 = self.enc_layer_1.encode(x, x_mask)
        enc_h2 = self.enc_layer_2.encode(enc_h1, x_mask)
        enc_h3 = self.enc_layer_3.encode(enc_h2, x_mask)
        enc_h4 = self.enc_layer_4.encode(enc_h3, x_mask)

        enc_h = tensor.concatenate(enc_h4, axis=1)

        context_vector = 0
        init_val = 0
        max_length = 100

        y = theano.scan(name          = 'seq2seq',
                        fn            = self.decoder_step(),
                        outputs_info  = init_val,
                        non_sequences = context_vector,
                        n_steps       = max_length)
































