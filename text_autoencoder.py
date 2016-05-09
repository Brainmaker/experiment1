#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  ------ Text Autoencoder ------
  Created by Xiaolin Wan, 5.9.2016

  Build a deeply document autoencoder with Bi-LSTM.
  We adopt a Bi-LSTM structure with four layer for encoding and four layer for decoding,
  a deeply structure is planted between encoder and decoder to extract high-level sentence representation.

  Draft* Do NOT cite.
"""

import sys
import time
import json

import numpy
import theano
import theano.tensor as tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams

epsilon = 1e-6
trng = MRG_RandomStreams(seed=888)
dtype = theano.config.floatX

def data2npfloatX(data):
    return numpy.asarray(data, dtype=dtype)


class Core:
    def __init__(self, name):
        self.params = {}.fromkeys(name)

    @staticmethod
    def get_random_weights(dim1, dim2, name=None):
        w = numpy.random.randn(dim1, dim2)
        return theano.shared(w.astype(dtype), name=name)

    @staticmethod
    def get_zero_bias(dim, name=None):
        return theano.shared(numpy.zeros((dim,)).astype(dtype), name=name)

    @staticmethod
    def get_4_ortho_weights(dim, name=None):
        u, s, v = numpy.linalg.svd(numpy.random.randn(dim, dim))
        ortho_weight = numpy.concatenate([u, u, u, u], axis=1)
        return theano.shared(ortho_weight.astype(dtype), name=name)

    @staticmethod
    def slice_4(x, slice_tag, dim):
        if x.ndim == 3:
            return x[:, :, slice_tag * dim: (slice_tag + 1) * dim]
        return x[:, slice_tag * dim: (slice_tag + 1) * dim]

    @staticmethod
    def dropout(state_before, trng_=trng):
        proj = tensor.switch(cond = theano.shared(data2npfloatX(0.)),
                             ift  = (state_before * trng_.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)),
                             iff  = state_before * 0.5)
        return proj

    def params2json(self):
        params = {}.fromkeys(self.params.keys())
        for kk in params:
            params[kk] = self.params[kk].tolist()
        return json.dumps(params)

    def json2params(self, jsonstr):
        params = json.loads(jsonstr)
        assert params.keys() == self.params.keys()
        for kk in params:
            self.params[kk] = theano.shared(numpy.array(params[kk], dtype=dtype), name=kk)


class Dense(Core):
    def __init__(self, dim1, dim2):
        Core.__init__(self, ['W', 'b'])
        self.dim1 = dim1
        self.dim2 = dim2
        self.params['W'] = Core.get_random_weights(self.dim1, self.dim2, name='W')
        self.params['b'] = Core.get_zero_bias(self.dim2, name='b')

    def forward(self, state_below):
        return tensor.dot(state_below, self.params['W']) + self.params['b']


class WordEmbeddingLayer(Core):
    def __init__(self, embedding_dim, vocab_size):
        Core.__init__(self, ['E'])
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.params['E'] = Core.get_random_weights(vocab_size, embedding_dim, name='E')

    def forward(self, one_hot_input):
        return tensor.dot(one_hot_input, self.params['E'])


class OutputLayer(Core):
    def __init__(self, dec_dim, vocab_size):
        Core.__init__(self, ['U0','E0','C0','W0'])
        self.dec_dim = dec_dim
        self.vocab_size = vocab_size
        self.params['U0'] = Core.get_random_weights(self.dec_dim, self.dec_dim)
        self.params['E0'] = Core.get_random_weights(self.vocab_size, self.dec_dim)
        self.params['C0'] = Core.get_random_weights(self.dec_dim, self.dec_dim)
        self.params['W0'] = Core.get_random_weights(self.dec_dim, self.vocab_size)

    def forward(self, y_tm1, h, context_vector):
        t_bar = (tensor.dot(h, self.params['U0']) +
                 tensor.dot(y_tm1, self.params['E0']) +
                 tensor.dot(context_vector, self.params['C0']))
        t = t_bar  # TODO: 2-maxout layer, the size of t_bar should be 2*dec_dim, here we use size(t_bar) = dec_dim
        prob_y = tensor.nnet.softmax(tensor.dot(t, self.params['W0']))
        y = prob_y  # TODO: argmax
        return prob_y, y


class BiLSTMEncodeLayer(Core):
    def __init__(self, dim):
        Core.__init__(self, ['f_W','f_U','f_V','f_b',
                             'b_W','b_U','b_V','b_b'])
        self.dim = dim
        self.params['f_W'] = Core.get_4_ortho_weights(self.dim, name='f_W')
        self.params['f_U'] = Core.get_4_ortho_weights(self.dim, name='f_U')
        self.params['f_V'] = Core.get_4_ortho_weights(self.dim, name='f_V')
        self.params['f_b'] = Core.get_zero_bias(4 * self.dim, name='f_b')

        self.params['b_W'] = Core.get_4_ortho_weights(self.dim, name='b_W')
        self.params['b_U'] = Core.get_4_ortho_weights(self.dim, name='b_U')
        self.params['b_V'] = Core.get_4_ortho_weights(self.dim, name='b_V')
        self.params['b_b'] = Core.get_zero_bias(4 * self.dim, name='b_b')

    def f_step(self, x, mask, h_tm1, c_tm1):
        preact = tensor.dot(h_tm1, self.params['f_U'])
        preact += x

        i = tensor.nnet.sigmoid(Core.slice_4(preact, 0, self.dim))
        f = tensor.nnet.sigmoid(Core.slice_4(preact, 1, self.dim))
        o = tensor.nnet.sigmoid(Core.slice_4(preact, 2, self.dim))
        c = tensor.tanh(Core.slice_4(preact, 3, self.dim))

        c = f * c_tm1 + i * c
        c = mask[:, None] * c + (1. - mask)[:, None] * c_tm1

        h = o * tensor.tanh(c)
        h = mask[:, None] * h + (1. - mask)[:, None] * h_tm1

        return h, c

    def b_step(self, x, mask, h_tm1, c_tm1):
        preact = tensor.dot(h_tm1, self.params['b_U'])
        preact += x

        i = tensor.nnet.sigmoid(Core.slice_4(preact, 0, self.dim))
        f = tensor.nnet.sigmoid(Core.slice_4(preact, 1, self.dim))
        o = tensor.nnet.sigmoid(Core.slice_4(preact, 2, self.dim))
        c = tensor.tanh(Core.slice_4(preact, 3, self.dim))

        c = f * c_tm1 + i * c
        c = mask[:, None] * c + (1. - mask)[:, None] * c_tm1

        h = o * tensor.tanh(c)
        h = mask[:, None] * h + (1. - mask)[:, None] * h_tm1

        return h, c

    def forward(self, emb_x, emb_rx, fstate_below, bstate_below, x_mask, rx_mask):
        nsteps = emb_x.shape[0]
        if emb_x.ndim == 3:
            n_samples = emb_x.shape[1]
        else:
            n_samples = 1

        init_val = [tensor.alloc(data2npfloatX(0.), n_samples, self.dim),
                    tensor.alloc(data2npfloatX(0.), n_samples, self.dim)]

        fstate = (tensor.dot(fstate_below, self.params['f_W']) +
                  tensor.dot(emb_x, self.params['f_V']) +
                  self.params['f_b'])

        bstate = (tensor.dot(bstate_below, self.params['b_W']) +
                  tensor.dot(emb_x, self.params['b_V']) +
                  self.params['b_b'])

        fstate_current, update = theano.scan(name          = 'LSTMForwardEncoder',
                                             fn            = self.f_step,
                                             sequences     = [fstate, x_mask],
                                             outputs_info  = init_val,
                                             non_sequences = None,
                                             n_steps       = nsteps)

        bstate_current, update = theano.scan(name          = 'LSTMBackwardEncoder',
                                             fn            = self.f_step,
                                             sequences     = [fstate, x_mask],
                                             outputs_info  = init_val,
                                             non_sequences = None,
                                             n_steps       = nsteps)

        return fstate_current[0], bstate_current[0]


class LSTMDecodeLayer(Core):
    def __init__(self, dim):
        Core.__init__(self, ['W','U','V','C','b'])
        self.dim = dim
        self.params['W'] = Core.get_4_ortho_weights(self.dim, name='W')
        self.params['U'] = Core.get_4_ortho_weights(self.dim, name='U')
        self.params['V'] = Core.get_4_ortho_weights(self.dim, name='V')
        self.params['C'] = Core.get_4_ortho_weights(self.dim, name='C')
        self.params['b'] = Core.get_zero_bias(4 * self.dim, name='b')

    def forward(self, state_below, emb_y_tm1, h_tm1, c_tm1, context_vector):
        preact = (tensor.dot(state_below, self.params['W']) +
                  tensor.dot(emb_y_tm1, self.params['V']) +
                  tensor.dot(h_tm1, self.params['U']) +
                  tensor.dot(context_vector, self.params['C']) +
                  self.params['b'])

        i = tensor.nnet.sigmoid(Core.slice_4(preact, 0, self.dim))
        f = tensor.nnet.sigmoid(Core.slice_4(preact, 1, self.dim))
        o = tensor.nnet.sigmoid(Core.slice_4(preact, 2, self.dim))
        c = tensor.tanh(Core.slice_4(preact, 3, self.dim))

        c = f * c_tm1 + i * c
        h = o * tensor.tanh(c)

        return h, c


class SubModule:
    def __init__(self, layers_name):
        self.layers = {}.fromkeys(layers_name)



    def layers2json(self):
        container = {}.fromkeys(self.layers.keys())
        for name in self.layers:
            container[name] = self.layers[name].params2json()
        return json.dumps(container)

    def json2layers(self, jsonstr):
        container = json.loads(jsonstr)
        assert container.keys() == self.layers.keys()
        for name in container:
            self.layers[name].json2params(container[name])


class Encoder(SubModule):
    def __init__(self, enc_dim, use_dropout=False):
        SubModule.__init__(self, ['enc_1','enc_2','enc_3','enc_4'])
        self.use_dropout = use_dropout
        self.enc_dim = enc_dim
        self.layers['enc_1'] = BiLSTMEncodeLayer(self.enc_dim)
        self.layers['enc_2'] = BiLSTMEncodeLayer(self.enc_dim)
        self.layers['enc_3'] = BiLSTMEncodeLayer(self.enc_dim)
        self.layers['enc_4'] = BiLSTMEncodeLayer(self.enc_dim)

    def encode(self, emb_x, x_mask):
        emb_rx = emb_x
        rx_mask = x_mask #TODO

        if self.use_dropout:
            fh1, bh1 = self.layers['enc_1'].forward(emb_x, emb_rx, emb_x, emb_rx, x_mask, rx_mask)
            fproj1, bproj1 = Core.dropout(fh1), Core.dropout(bh1)
            fh2, bh2 = self.layers['enc_2'].forward(emb_x, emb_rx, fproj1, bproj1, x_mask, rx_mask)
            fproj2, bproj2 = Core.dropout(fh2), Core.dropout(bh2)
            fh3, bh3 = self.layers['enc_3'].forward(emb_x, emb_rx, fproj2, bproj2, x_mask, rx_mask)
            fproj3, bproj3 = Core.dropout(fh3), Core.dropout(bh3)
            fh4, bh4 = self.layers['enc_4'].forward(emb_x, emb_rx, fproj3, bproj3, x_mask, rx_mask)
        else:
            fh1, bh1 = self.layers['enc_1'].forward(emb_x, emb_rx, emb_x, emb_rx, x_mask, rx_mask)
            fh2, bh2 = self.layers['enc_2'].forward(emb_x, emb_rx, fh1, bh1, x_mask, rx_mask)
            fh3, bh3 = self.layers['enc_3'].forward(emb_x, emb_rx, fh2, bh2, x_mask, rx_mask)
            fh4, bh4 = self.layers['enc_4'].forward(emb_x, emb_rx, fh3, bh3, x_mask, rx_mask)

        return tensor.concatenate([fh4, bh4], axis=1)


class Decoder(SubModule):
    def __init__(self, dec_dim, dec_nsteps, use_dropout):
        SubModule.__init__(self, ['dec_1','dec_2','dec_3','dec_4'])
        self.use_dropout = use_dropout
        self.dec_dim = dec_dim
        self.dec_nsteps = dec_nsteps
        self.layers['dec_1'] = LSTMDecodeLayer(self.dec_dim)
        self.layers['dec_2'] = LSTMDecodeLayer(self.dec_dim)
        self.layers['dec_3'] = LSTMDecodeLayer(self.dec_dim)
        self.layers['dec_4'] = LSTMDecodeLayer(self.dec_dim)

    def _get_init_state(self, batch_size):
        init_state = []
        for i in range(0, 9):
            init_state.append(tensor.alloc(data2npfloatX(0.), batch_size, self.dec_dim))
        return init_state

    def _step(self, emb_y_tm1,
              c1_tm1, c2_tm1, c3_tm1, c4_tm1,
              h1_tm1, h2_tm1, h3_tm1, h4_tm1,
              context_vector):
        if self.use_dropout:
            h1, c1 = self.layers['dec_1'].forward(emb_y_tm1, emb_y_tm1, h1_tm1, c1_tm1, context_vector)
            proj1 = Core.dropout(h1)
            h2, c2 = self.layers['dec_2'].forward(proj1, emb_y_tm1, h2_tm1, c2_tm1, context_vector)
            proj2 = Core.dropout(h2)
            h3, c3 = self.layers['dec_3'].forward(proj2, emb_y_tm1, h3_tm1, c3_tm1, context_vector)
            proj3 = Core.dropout(h3)
            h4, c4 = self.layers['dec_4'].forward(proj3, emb_y_tm1, h4_tm1, c4_tm1, context_vector)
        else:
            h1, c1 = self.layers['dec_1'].forward(emb_y_tm1, emb_y_tm1, h1_tm1, c1_tm1, context_vector)
            h2, c2 = self.layers['dec_2'].forward(h1, emb_y_tm1, h2_tm1, c2_tm1, context_vector)
            h3, c3 = self.layers['dec_3'].forward(h2, emb_y_tm1, h3_tm1, c3_tm1, context_vector)
            h4, c4 = self.layers['dec_4'].forward(h3, emb_y_tm1, h4_tm1, c4_tm1, context_vector)
        emb_y = h4
        return (emb_y,
                c1, c2, c3, c4,
                h1, h2, h3, h4)


    def decode(self, context_vector):
        batch_size = context_vector.shape[0]
        init_state = self._get_init_state(batch_size)
        rval, updates = theano.scan(name          = 'Decoder',
                                    fn            = self._step,
                                    outputs_info  = init_state,
                                    non_sequences = [context_vector],
                                    n_steps       = self.dec_nsteps)


class WordEncoder(Encoder):
    def __init__(self, enc_dim, vocab_size, use_dropout):
        Encoder.__init__(self, enc_dim, use_dropout)
        self.vocab_dim = vocab_size
        self.layers['word_embedding'] = WordEmbeddingLayer(enc_dim, vocab_size)

    def word_encode(self, x, x_mask):
        if self.use_dropout:
            _emb_x = self.layers['word_embedding'].forward(x)
            emb_x = Core.dropout(_emb_x)
        else:
            emb_x = self.layers['word_embedding'].forward(x)
        return self.encode(emb_x, x_mask)


class SentEncoder(Encoder):
    def __init__(self, enc_dim, use_dropout):
        Encoder.__init__(self, enc_dim, use_dropout)


class WordDecoder(Decoder):
    def __init__(self, dec_dim, vocab_size, dec_nsteps, use_dropout, word_emb_layer):
        Decoder.__init__(dec_dim, dec_nsteps, use_dropout)
        self.vocab_size = vocab_size
        self.layers['output_layer'] = OutputLayer(self.dec_dim, self.vocab_size)
        self.word_emb_layer = word_emb_layer

    def _step(self, y_tm1,
              c1_tm1, c2_tm1, c3_tm1, c4_tm1,
              h1_tm1, h2_tm1, h3_tm1, h4_tm1,
              context_vector):
        emb_y_tm1 = self.word_emb_layer.forward(y_tm1)
        
        if self.use_dropout:
            h1, c1 = self.layers['dec_1'].forward(emb_y_tm1, emb_y_tm1, h1_tm1, c1_tm1, context_vector)
            proj1 = Core.dropout(h1)
            h2, c2 = self.layers['dec_2'].forward(proj1, emb_y_tm1, h2_tm1, c2_tm1, context_vector)
            proj2 = Core.dropout(h2)
            h3, c3 = self.layers['dec_3'].forward(proj2, emb_y_tm1, h3_tm1, c3_tm1, context_vector)
            proj3 = Core.dropout(h3)
            h4, c4 = self.layers['dec_4'].forward(proj3, emb_y_tm1, h4_tm1, c4_tm1, context_vector)
        else:
            h1, c1 = self.layers['dec_1'].forward(emb_y_tm1, emb_y_tm1, h1_tm1, c1_tm1, context_vector)
            h2, c2 = self.layers['dec_2'].forward(h1, emb_y_tm1, h2_tm1, c2_tm1, context_vector)
            h3, c3 = self.layers['dec_3'].forward(h2, emb_y_tm1, h3_tm1, c3_tm1, context_vector)
            h4, c4 = self.layers['dec_4'].forward(h3, emb_y_tm1, h4_tm1, c4_tm1, context_vector)

        y = self.layers['output_layer'].forward(y_tm1, h4, context_vector)
        return (y,
                c1, c2, c3, c4,
                h1, h2, h3, h4)


class SentDecoder(Decoder):
    def __init__(self, dec_dim, dec_nsteps, use_dropout):
        Decoder.__init__(self, dec_dim, dec_nsteps, use_dropout)


class SAE:
    pass


class DAE:
    pass
