#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from collections import OrderedDict

import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

EPS = 1e-6
TRNG = MRG_RandomStreams(seed=888)
DTYPE = theano.config.floatX
IDX_TYPE = 'int64'


def dtype_cast(data):
    return numpy.asarray(data, dtype=DTYPE)


def unzip2list(zipped):
    pass


def dropout(state_before):
    proj = tensor.switch(theano.shared(dtype_cast(0.)),
                         state_before * TRNG.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                         state_before * 0.5)
    return proj


def _get_random_weights(dim1, dim2, name=None):
    w = numpy.random.randn(dim1, dim2)
    return theano.shared(w.astype(DTYPE), name=name)


def _get_zero_bias(dim, name=None):
    return theano.shared(numpy.zeros((dim,)).astype(DTYPE), name=name)


def _get_4_ortho_weights(dim, name=None):
    u, s, v = numpy.linalg.svd(numpy.random.randn(dim, dim))
    ortho_weight = numpy.concatenate([u, u, u, u], axis=1)
    return theano.shared(ortho_weight.astype(DTYPE), name=name)


def _slice_4(x, slice_tag, dim):
    if x.ndim == 3:
        return x[:, :, slice_tag * dim: (slice_tag + 1) * dim]
    return x[:, slice_tag * dim: (slice_tag + 1) * dim]


class Core(object):
    def __init__(self, name):
        self.tparams = OrderedDict().fromkeys(name)

    def get_params(self):
        return [p for p in self.tparams.values()]

    def view_params(self):
        return [p.get_value() for p in self.tparams.values()]

    def params2json(self):
        tparams = OrderedDict().fromkeys(self.tparams.keys())
        for kk in tparams.keys():
            tparams[kk] = self.tparams[kk].get_value().tolist()
        return json.dumps(tparams)

    def json2params(self, jsonstr):
        tparams = json.loads(jsonstr)
        assert tparams.keys() == self.tparams.keys()
        for kk in tparams:
            self.tparams[kk] = theano.shared(numpy.array(tparams[kk], dtype=DTYPE), name=kk)


class Dense(Core):
    def __init__(self, dim1, dim2):
        Core.__init__(self, ['W', 'b'])
        self.dim1 = dim1
        self.dim2 = dim2
        self.tparams['W'] = _get_random_weights(self.dim1, self.dim2, name='W')
        self.tparams['b'] = _get_zero_bias(self.dim2, name='b')

    def forward(self, state_below):
        return tensor.dot(state_below, self.tparams['W']) + self.tparams['b']


class WordEmbeddingLayer(Core):
    def __init__(self, embedding_dim, vocab_size):
        Core.__init__(self, ['E'])
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.tparams['E'] = _get_random_weights(vocab_size, embedding_dim, name='E')

    def forward(self, one_hot_input):
        return tensor.dot(one_hot_input, self.tparams['E'])


class OutputLayer(Core):
    def __init__(self, dec_dim, vocab_size):
        Core.__init__(self, ['U0', 'E0', 'C0', 'W0'])
        self.dec_dim = dec_dim
        self.vocab_size = vocab_size
        self.tparams['U0'] = _get_random_weights(self.dec_dim, self.dec_dim, name='U0')
        self.tparams['E0'] = _get_random_weights(self.vocab_size, self.dec_dim, name='E0')
        self.tparams['C0'] = _get_random_weights(self.dec_dim, self.dec_dim, name='C0')
        self.tparams['W0'] = _get_random_weights(self.dec_dim, self.vocab_size, name='W0')

    def forward(self, y_tm1, h, context_vector):
        t_bar = (tensor.dot(h, self.tparams['U0']) +
                 tensor.dot(y_tm1, self.tparams['E0']) +
                 tensor.dot(context_vector, self.tparams['C0']))
        t = t_bar  # TODO: 2-maxout layer, the size of t_bar should be 2*dec_dim, here we use size(t_bar) = dec_dim
        prob_y = tensor.nnet.softmax(tensor.dot(t, self.tparams['W0']))
        y_idx = tensor.argmax(prob_y, axis=1)
        y = theano.tensor.extra_ops.to_one_hot(y_idx, self.vocab_size, dtype=DTYPE)
        return y, prob_y


class BiLSTMEncodeLayer(Core):
    def __init__(self, dim):
        Core.__init__(self, ['f_W', 'f_U', 'f_V', 'f_b',
                             'b_W', 'b_U', 'b_V', 'b_b'])
        self.dim = dim
        self.tparams['f_W'] = _get_4_ortho_weights(self.dim, name='f_W')
        self.tparams['f_U'] = _get_4_ortho_weights(self.dim, name='f_U')
        self.tparams['f_V'] = _get_4_ortho_weights(self.dim, name='f_V')
        self.tparams['f_b'] = _get_zero_bias(4 * self.dim, name='f_b')

        self.tparams['b_W'] = _get_4_ortho_weights(self.dim, name='b_W')
        self.tparams['b_U'] = _get_4_ortho_weights(self.dim, name='b_U')
        self.tparams['b_V'] = _get_4_ortho_weights(self.dim, name='b_V')
        self.tparams['b_b'] = _get_zero_bias(4 * self.dim, name='b_b')

    def f_step(self, x, mask, h_tm1, c_tm1):
        preact = tensor.dot(h_tm1, self.tparams['f_U'])
        preact += x

        i = tensor.nnet.sigmoid(_slice_4(preact, 0, self.dim))
        f = tensor.nnet.sigmoid(_slice_4(preact, 1, self.dim))
        o = tensor.nnet.sigmoid(_slice_4(preact, 2, self.dim))
        c = tensor.tanh(_slice_4(preact, 3, self.dim))

        c = f * c_tm1 + i * c
        c = mask[:, None] * c + (1. - mask)[:, None] * c_tm1

        h = o * tensor.tanh(c)
        h = mask[:, None] * h + (1. - mask)[:, None] * h_tm1

        return h, c

    def b_step(self, x, mask, h_tm1, c_tm1):
        preact = tensor.dot(h_tm1, self.tparams['b_U'])
        preact += x

        i = tensor.nnet.sigmoid(_slice_4(preact, 0, self.dim))
        f = tensor.nnet.sigmoid(_slice_4(preact, 1, self.dim))
        o = tensor.nnet.sigmoid(_slice_4(preact, 2, self.dim))
        c = tensor.tanh(_slice_4(preact, 3, self.dim))

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

        init_val = [tensor.alloc(dtype_cast(0.), n_samples, self.dim),
                    tensor.alloc(dtype_cast(0.), n_samples, self.dim)]

        fstate = (tensor.dot(fstate_below, self.tparams['f_W']) +
                  tensor.dot(emb_x, self.tparams['f_V']) +
                  self.tparams['f_b'])

        bstate = (tensor.dot(bstate_below, self.tparams['b_W']) +
                  tensor.dot(emb_rx, self.tparams['b_V']) +
                  self.tparams['b_b'])

        fstate_current, _ = theano.scan(
            name          = 'LSTMForwardEncoder',
            fn            = self.f_step,
            sequences     = [fstate, x_mask],
            outputs_info  = init_val,
            non_sequences = None,
            n_steps       = nsteps
        )

        bstate_current, _ = theano.scan(
            name          = 'LSTMBackwardEncoder',
            fn            = self.b_step,
            sequences     = [bstate, rx_mask],
            outputs_info  = init_val,
            non_sequences = None,
            n_steps       = nsteps
        )

        return fstate_current[0], bstate_current[0]


class LSTMDecodeLayer(Core):
    def __init__(self, dim):
        Core.__init__(self, ['W', 'U', 'V', 'C', 'b'])
        self.dim = dim
        self.tparams['W'] = _get_4_ortho_weights(self.dim, name='W')
        self.tparams['U'] = _get_4_ortho_weights(self.dim, name='U')
        self.tparams['V'] = _get_4_ortho_weights(self.dim, name='V')
        self.tparams['C'] = _get_4_ortho_weights(self.dim, name='C')
        self.tparams['b'] = _get_zero_bias(4 * self.dim, name='b')

    def forward(self, state_below, emb_y_tm1, mask, h_tm1, c_tm1, context_vector):
        preact = (tensor.dot(state_below, self.tparams['W']) +
                  tensor.dot(emb_y_tm1, self.tparams['V']) +
                  tensor.dot(h_tm1, self.tparams['U']) +
                  tensor.dot(context_vector, self.tparams['C']) +
                  self.tparams['b'])

        i = tensor.nnet.sigmoid(_slice_4(preact, 0, self.dim))
        f = tensor.nnet.sigmoid(_slice_4(preact, 1, self.dim))
        o = tensor.nnet.sigmoid(_slice_4(preact, 2, self.dim))
        c = tensor.tanh(_slice_4(preact, 3, self.dim))

        c = f * c_tm1 + i * c
        c = mask[:, None] * c + (1. - mask)[:, None] * c_tm1
        h = o * tensor.tanh(c)
        h = mask[:, None] * h + (1. - mask)[:, None] * h_tm1

        return h, c
