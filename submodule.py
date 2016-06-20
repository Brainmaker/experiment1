#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
from collections import OrderedDict

import theano
import theano.tensor as tensor

from core import (
    IDX_TYPE, DTYPE, dtype_cast, dropout, dropout_2, Dense, WordEncodeLayer, OutputLayer, BiLSTMEncodeLayer, LSTMDecodeLayer)


class SubModule(object):
    def __init__(self, layers_name):
        self.layers = OrderedDict().fromkeys(layers_name)

    def get_params(self):
        module_params = []
        for kk in self.layers.keys():
            module_params.extend(self.layers[kk].get_params())
        return module_params

    def layers2json(self):
        container = OrderedDict().fromkeys(self.layers.keys())
        for name in self.layers.keys():
            container[name] = self.layers[name].params2json()
        return json.dumps(container)

    def json2layers(self, jsonstr):
        container = json.loads(jsonstr)
        assert container.keys() == self.layers.keys()
        for name in container:
            self.layers[name].json2params(container[name])


class Encoder(SubModule):
    def __init__(self, enc_dim):
        SubModule.__init__(self, ['enc_1', 'enc_2', 'enc_3'])
        self.enc_dim = enc_dim
        self.layers['enc_1'] = BiLSTMEncodeLayer(self.enc_dim)
        self.layers['enc_2'] = BiLSTMEncodeLayer(self.enc_dim)
        self.layers['enc_3'] = BiLSTMEncodeLayer(self.enc_dim)

    def encode(self, emb_x, x_mask):
        emb_rx = emb_x[:, ::-1]  # reverse
        rx_mask = x_mask[:, ::-1]
        fh1, bh1 = self.layers['enc_1'].forward(emb_x, emb_rx, emb_x, emb_rx, x_mask, rx_mask)
        fh2, bh2 = self.layers['enc_2'].forward(emb_x, emb_rx, fh1, bh1, x_mask, rx_mask)
        fh3, bh3 = self.layers['enc_3'].forward(emb_x, emb_rx, fh2, bh2, x_mask, rx_mask)
        return tensor.concatenate([fh3[0], bh3[0]], axis=1)


class WordEncoder(Encoder):
    def __init__(self, enc_dim, emb_size, use_dropout):
        Encoder.__init__(self, enc_dim)
        self.use_dropout = use_dropout
        self.emb_dim = emb_size
        self.layers['word_embedding'] = WordEncodeLayer(enc_dim, emb_size)

    def word_encode(self, x, x_mask):
        if self.use_dropout:
            _emb_x = self.layers['word_embedding'].forward(x)
            emb_x = dropout(_emb_x)
        else:
            emb_x = self.layers['word_embedding'].forward(x)
        return self.encode(emb_x, x_mask)


class SentEncoder(Encoder):
    def __init__(self, enc_dim, use_dropout):
        Encoder.__init__(self, enc_dim)
        self.use_dropout = use_dropout

    def sent_encode(self, x, x_mask):
        if self.use_dropout:#TODO:
            x_ = dropout(x)
        else:
            x_ = x
        return self.encode(x_, x_mask)


class Decoder(SubModule):
    def __init__(self, dec_dim, use_dropout):
        SubModule.__init__(self, ['dec_1', 'dec_2', 'dec_3'])
        self.use_dropout = use_dropout
        self.dec_dim = dec_dim
        self.layers['dec_1'] = LSTMDecodeLayer(self.dec_dim)
        self.layers['dec_2'] = LSTMDecodeLayer(self.dec_dim)
        self.layers['dec_3'] = LSTMDecodeLayer(self.dec_dim)

    def _get_init_state(self, batch_size):
        return [tensor.alloc(dtype_cast(0.), batch_size, self.dec_dim) for i in range(8)]

    def _step(self, mask, y_tm1, _, c1_tm1, c2_tm1, c3_tm1, h1_tm1, h2_tm1, h3_tm1, context_vector):

        emb_y_tm1 = y_tm1
        batch_size = context_vector.shape[0]
        shape = (batch_size, self.dec_dim)
        # TODO: dropout
        h1, c1 = self.layers['dec_1'].forward(emb_y_tm1, emb_y_tm1, mask, h1_tm1, c1_tm1, context_vector)
        h2, c2 = self.layers['dec_2'].forward(h1, emb_y_tm1, mask, h2_tm1, c2_tm1, context_vector)
        h3, c3 = self.layers['dec_3'].forward(h2, emb_y_tm1, mask, h3_tm1, c3_tm1, context_vector)
        y = h3
        return y, _, c1, c2, c3, h1, h2, h3

    def decode(self, context_vector, target_seq_mask):
        batch_size = context_vector.shape[0]
        init_state = self._get_init_state(batch_size)
        rval, _ = theano.scan(
            name          = 'Decoder',
            fn            = self._step,
            sequences     = target_seq_mask,
            outputs_info  = init_state,
            non_sequences = context_vector
        )

        return rval[0], rval[1]


class WordDecoder(Decoder):
    def __init__(self, dec_dim, emb_size, vocab_size, word_emb_layer, word_table, use_dropout):
        Decoder.__init__(self, dec_dim, use_dropout)
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.layers['output_layer'] = OutputLayer(self.dec_dim, self.emb_size, self.vocab_size, word_table)
        self.word_emb_layer = word_emb_layer

    def _get_init_state(self, batch_size):
        init_state = [tensor.alloc(dtype_cast(0.), batch_size, self.emb_size),
                      tensor.alloc(dtype_cast(0.), batch_size, self.vocab_size)]
        init_state += [tensor.alloc(dtype_cast(0.), batch_size, self.dec_dim) for i in range(6)]
        return init_state

    def _step(self, mask, y_tm1, _, c1_tm1, c2_tm1, c3_tm1, h1_tm1, h2_tm1, h3_tm1, context_vector):
        batch_size = context_vector.shape[0]
        shape = (batch_size, self.dec_dim)
        if self.use_dropout:
            emb_y_tm1 = dropout_2(self.word_emb_layer.forward(y_tm1), shape)
            h1, c1 = self.layers['dec_1'].forward(emb_y_tm1, emb_y_tm1, mask, h1_tm1, c1_tm1, context_vector)
            h2, c2 = self.layers['dec_2'].forward(h1, emb_y_tm1, mask, h2_tm1, c2_tm1, context_vector)
            h3, c3 = self.layers['dec_3'].forward(h2, emb_y_tm1, mask, h3_tm1, c3_tm1, context_vector)
            y, prob_y = self.layers['output_layer'].forward(y_tm1, dropout_2(h3, shape), context_vector)
        else:
            emb_y_tm1 = self.word_emb_layer.forward(y_tm1), shape
            h1, c1 = self.layers['dec_1'].forward(emb_y_tm1, emb_y_tm1, mask, h1_tm1, c1_tm1, context_vector)
            h2, c2 = self.layers['dec_2'].forward(h1, emb_y_tm1, mask, h2_tm1, c2_tm1, context_vector)
            h3, c3 = self.layers['dec_3'].forward(h2, emb_y_tm1, mask, h3_tm1, c3_tm1, context_vector)
            y, prob_y = self.layers['output_layer'].forward(y_tm1, h3, context_vector)

        return y, prob_y, c1, c2, c3, h1, h2, h3


class SentDecoder(Decoder):
    def __init__(self, dec_dim, use_dropout):
        Decoder.__init__(self, dec_dim, use_dropout)


class MLP(SubModule):
    def __init__(self, input_dim, output_dim, use_dropout):
        SubModule.__init__(self, ['layer_1', 'layer_2', 'layer_3'])
        self.use_dropout = use_dropout
        assert 2 * output_dim == input_dim
        mid_dim_1 = int(1.67 * output_dim)
        mid_dim_2 = int(1.33 * output_dim)

        self.layers['layer_1'] = Dense(input_dim, mid_dim_1)
        self.layers['layer_2'] = Dense(mid_dim_1, mid_dim_2)
        self.layers['layer_3'] = Dense(mid_dim_2, output_dim)

    def forward(self, x):
        if self.use_dropout:
            h1 = self.layers['layer_1'].forward(dropout(x))
            h2 = self.layers['layer_2'].forward(h1)
            h3_ = self.layers['layer_3'].forward(h2)
            h3 = dropout(h3_)
        else:
            h1 = self.layers['layer_1'].forward(x)
            h2 = self.layers['layer_2'].forward(h1)
            h3 = self.layers['layer_3'].forward(h2)

        return h3
