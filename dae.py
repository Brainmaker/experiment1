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


def data2npfloatX(data):
    return numpy.asarray(data, dtype=dtype)


def get_4_ortho_weight(dim):
    """
    Get a (n * 4n) matrix
    """
    tmp = numpy.random.randn(dim, dim)
    u, s, v = numpy.linalg.svd(tmp)
    u = u.astype(config.floatX)
    ortho_weight = numpy.concatenate([u, u, u, u], axis=1)
    return ortho_weight


def get_random_weights(dim1, dim2):
    w = numpy.random.randn(dim1, dim2)
    return w.astype(dtype)


def tensor_slice(x, slice_tag, dim):
    """
    使用注意！例如W的大小是(N * 4N)，则dim=N
    """
    if x.ndim == 3:
        return x[:, :, slice_tag*dim : (slice_tag+1)*dim]
    return x[:, slice_tag*dim : (slice_tag+1)*dim]


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


class Encoder:

    class LSTMEncoderLayer:
        def __init__(self, dim):
            # W, U is (n * 4n), b is (1 * 4n)
            self.dim = dim
            self.nn_weights = {}.fromkeys(['W', 'U', 'b'])

            self.nn_weights['W'] = np2shared(get_4_ortho_weight(dim), 'W')
            self.nn_weights['U'] = np2shared(get_4_ortho_weight(dim), 'U')
            self.nn_weights['b'] = np2shared(numpy.zeros((4 * dim,)).astype(dtype), 'b')


        def encode_step(self, x, mask, h_tm1, c_tm1):
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
            c = mask[:, None] * c + (1. - mask)[:, None] * c_tm1

            h = o * tensor.tanh(c)
            h = mask[:, None] * h + (1. - mask)[:, None] * h_tm1

            return h, c


        def layer_encode(self, x, x_mask):
            nsteps = x.shape[0]
            if x.ndim == 3:
                n_samples = x.shape[1]
            else:
                n_samples = 1

            init_val = [tensor.alloc(data2npfloatX(0.), n_samples, self.dim),
                        tensor.alloc(data2npfloatX(0.), n_samples, self.dim)]

            state = (tensor.dot(x, self.nn_weights['W']) +
                     self.nn_weights['b'])

            layer_output, update = theano.scan(name          = 'LSTMEncoder',
                                               fn            = self.encode_step,
                                               sequences     = [state, x_mask],
                                               outputs_info  = init_val,
                                               non_sequences = None,
                                               n_steps       = nsteps)
            return layer_output[0]


    class MLP:
        def __init__(self, dim1, dim2, dim3):
            self.dim1, self.dim2, self.dim3 = dim1, dim2, dim3
            self.nn_weights = {}.fromkeys(['W1','W2','W3','b1','b2','b3'])

            self.nn_weights['W1'] = np2shared(get_random_weights(dim1, dim2), 'W1')
            self.nn_weights['W2'] = np2shared(get_random_weights(dim2, dim2), 'W2')
            self.nn_weights['W3'] = np2shared(get_random_weights(dim2, dim3), 'W3')
            self.nn_weights['b1'] = np2shared(numpy.zeros((dim2,)).astype(dtype), 'b1')
            self.nn_weights['b2'] = np2shared(numpy.zeros((dim2,)).astype(dtype), 'b2')
            self.nn_weights['b3'] = np2shared(numpy.zeros((dim3,)).astype(dtype), 'b3')


        def mlp_encode(self, x):
            h1 = tensor.nnet.relu(tensor.dot(x, self.nn_weights['W1']) + self.nn_weights['b1'])
            h2 = tensor.nnet.relu(tensor.dot(h1, self.nn_weights['W2']) + self.nn_weights['b2'])
            h3 = tensor.nnet.relu(tensor.dot(h2, self.nn_weights['W3']) + self.nn_weights['b3'])
            return h3


    def __init__(self, enc_dim, mlp_dim1, mlp_dim2, mlp_dim3):
        self.enc_dim = enc_dim
        self.mlp_dim1 = mlp_dim1
        self.mlp_dim2 = mlp_dim2
        self.mlp_dim3 = mlp_dim3

        assert 2 * self.enc_dim == self.mlp_dim1
        
        self.fenc_1 = Encoder.LSTMEncoderLayer(self.enc_dim)
        self.fenc_2 = Encoder.LSTMEncoderLayer(self.enc_dim)
        self.fenc_3 = Encoder.LSTMEncoderLayer(self.enc_dim)
        self.fenc_4 = Encoder.LSTMEncoderLayer(self.enc_dim)
        self.benc_1 = Encoder.LSTMEncoderLayer(self.enc_dim)
        self.benc_2 = Encoder.LSTMEncoderLayer(self.enc_dim)
        self.benc_3 = Encoder.LSTMEncoderLayer(self.enc_dim)
        self.benc_4 = Encoder.LSTMEncoderLayer(self.enc_dim)
        self.mlp_enc = Encoder.MLP(self.mlp_dim1, self.mlp_dim2, self.mlp_dim3)


    def encode(self, x, x_mask):

        emb_x = x
        emb_rx = x
        rx_mask = x_mask

        fenc_h1 = self.fenc_1.layer_encode(emb_x, x_mask)
        fenc_h2 = self.fenc_2.layer_encode(fenc_h1, x_mask)
        fenc_h3 = self.fenc_3.layer_encode(fenc_h2, x_mask)
        fenc_h4 = self.fenc_4.layer_encode(fenc_h3, x_mask)

        benc_h1 = self.fenc_1.layer_encode(emb_rx, rx_mask)
        benc_h2 = self.fenc_2.layer_encode(benc_h1, rx_mask)
        benc_h3 = self.fenc_3.layer_encode(benc_h2, rx_mask)
        benc_h4 = self.fenc_4.layer_encode(benc_h3, rx_mask)

        enc_h = tensor.concatenate([fenc_h4, benc_h4], axis=1)

        context_vector = self.mlp_enc.mlp_encode(enc_h)

        return context_vector


class Decoder:

    class LSTMDecoderLayer:
        def __init__(self, dim):
            # W, U is (n * 4n), b is (1 * 4n)
            self.dim = dim
            self.nn_weights = {}.fromkeys(['W', 'U', 'C', 'b'])

            self.nn_weights['W'] = np2shared(get_4_ortho_weight(dim), 'W')
            self.nn_weights['U'] = np2shared(get_4_ortho_weight(dim), 'U')
            self.nn_weights['C'] = np2shared(get_4_ortho_weight(dim), 'C')
            self.nn_weights['b'] = np2shared(numpy.zeros((4 * dim,)).astype(dtype), 'b')


        def layer_decode_step(self, y_tm1, mask, h_tm1, c_tm1, context_vector):
            preact = (tensor.dot(y_tm1, self.nn_weights['W']) +
                      tensor.dot(h_tm1, self.nn_weights['U']) +
                      tensor.dot(context_vector, self.nn_weights['C']) +
                      self.nn_weights['b'])
            # preact += state
            print(mask)

            i = tensor.nnet.sigmoid(tensor_slice(preact, 0, self.dim))
            f = tensor.nnet.sigmoid(tensor_slice(preact, 1, self.dim))
            o = tensor.nnet.sigmoid(tensor_slice(preact, 2, self.dim))
            c = tensor.tanh(tensor_slice(preact, 3, self.dim))

            c = f * c_tm1 + i * c
            #c = mask[:, None] * c + (1. - mask)[:, None] * c_tm1

            h = o * tensor.tanh(c)
            #h = mask[:, None] * h + (1. - mask)[:, None] * h_tm1

            return h, c


    def __init__(self, dec_dim, dec_steps=100):
        self.dec_dim = dec_dim
        self.lstm_dec_n_layers = 4
        self.dec_steps = dec_steps

        self.dec_layer_1 = Decoder.LSTMDecoderLayer(self.dec_dim)
        self.dec_layer_2 = Decoder.LSTMDecoderLayer(self.dec_dim)
        self.dec_layer_3 = Decoder.LSTMDecoderLayer(self.dec_dim)
        self.dec_layer_4 = Decoder.LSTMDecoderLayer(self.dec_dim)


    def _get_init_state(self, batch_size):
        init_state = []
        for i in range(0, 9):
            init_state.append(tensor.alloc(data2npfloatX(0.), batch_size, self.dec_dim))

        return init_state


    def decode_step(self, y_tm1,
                    c1_tm1, c2_tm1, c3_tm1, c4_tm1,
                    h1_tm1, h2_tm1, h3_tm1, h4_tm1,
                    context_vector):
        mask = 0
        h1, c1 = self.dec_layer_1.layer_decode_step(y_tm1, mask, h1_tm1, c1_tm1, context_vector)
        h2, c2 = self.dec_layer_1.layer_decode_step(h1, mask, h2_tm1, c2_tm1, context_vector)
        h3, c3 = self.dec_layer_1.layer_decode_step(h2, mask, h3_tm1, c3_tm1, context_vector)
        h4, c4 = self.dec_layer_1.layer_decode_step(h3, mask, h4_tm1, c4_tm1, context_vector)

        y = h4 #TODO: convert to y
        return (y,
                c1, c2, c3, c4,
                h1, h2, h3, h4)


    def decode(self, context_vector):
        batch_size = context_vector.shape[0]
        init_state = self._get_init_state(batch_size)
        rval, updates = theano.scan(name          = 'seq2seq',
                                    fn            = self.decode_step,
                                    outputs_info  = init_state, #
                                    non_sequences = context_vector, # 4*3
                                    n_steps       = self.dec_steps)
        prob_pred_seq = rval[0]
        return prob_pred_seq


class Seq2Seq:
    def __init__(self, params):
        self.encoder = Encoder(enc_dim  = params['enc_dim'],
                               mlp_dim1 = params['mlp_dim1'],
                               mlp_dim2 = params['mlp_dim2'],
                               mlp_dim3 = params['mlp_dim3'])

        self.decoder = Decoder(dec_dim   = params['dec_dim'],
                               dec_steps = params['dec_steps'])


    def build_model(self, x, x_mask):
        context_vectors = self.encoder.encode(x, x_mask)
        prob_pred_seq = self.decoder.decode(context_vectors)
        pred_seq = prob_pred_seq #TODO
        return prob_pred_seq, pred_seq








































