#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from collections import OrderedDict

import theano
import theano.tensor as tensor

from core import \
    dtype_cast, dropout, Dense, WordEmbeddingLayer, OutputLayer, BiLSTMEncodeLayer, LSTMDecodeLayer

class SubModule(object):
    def __init__(self, layers_name):
        self.layers = OrderedDict().fromkeys(layers_name)

    def get_params(self):
        module_params = []
        for layer_params in self.layers.values().get_params():
            module_params.extend(layer_params)
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
    def __init__(self, enc_dim, use_dropout):
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
            fh2, bh2 = self.layers['enc_2'].forward(emb_x, emb_rx, dropout(fh1), dropout(bh1), x_mask, rx_mask)
            fh3, bh3 = self.layers['enc_3'].forward(emb_x, emb_rx, dropout(fh2), dropout(bh2), x_mask, rx_mask)
            fh4, bh4 = self.layers['enc_4'].forward(emb_x, emb_rx, dropout(fh3), dropout(bh3), x_mask, rx_mask)
        else:
            fh1, bh1 = self.layers['enc_1'].forward(emb_x, emb_rx, emb_x, emb_rx, x_mask, rx_mask)
            fh2, bh2 = self.layers['enc_2'].forward(emb_x, emb_rx, fh1, bh1, x_mask, rx_mask)
            fh3, bh3 = self.layers['enc_3'].forward(emb_x, emb_rx, fh2, bh2, x_mask, rx_mask)
            fh4, bh4 = self.layers['enc_4'].forward(emb_x, emb_rx, fh3, bh3, x_mask, rx_mask)

        return tensor.concatenate([fh4[0], bh4[0]], axis=1)


class WordEncoder(Encoder):
    def __init__(self, enc_dim, vocab_size, use_dropout=True):
        Encoder.__init__(self, enc_dim, use_dropout)
        self.vocab_dim = vocab_size
        self.layers['word_embedding'] = WordEmbeddingLayer(enc_dim, vocab_size)

    def word_encode(self, x, x_mask):
        if self.use_dropout:
            _emb_x = self.layers['word_embedding'].forward(x)
            emb_x = dropout(_emb_x)
        else:
            emb_x = self.layers['word_embedding'].forward(x)
        return self.encode(emb_x, x_mask)


class SentEncoder(Encoder):
    def __init__(self, enc_dim, use_dropout=True):
        Encoder.__init__(self, enc_dim, use_dropout)


class Decoder(SubModule):
    def __init__(self, dec_dim, use_dropout):
        SubModule.__init__(self, ['dec_1','dec_2','dec_3','dec_4'])
        self.use_dropout = use_dropout
        self.dec_dim = dec_dim
        self.layers['dec_1'] = LSTMDecodeLayer(self.dec_dim)
        self.layers['dec_2'] = LSTMDecodeLayer(self.dec_dim)
        self.layers['dec_3'] = LSTMDecodeLayer(self.dec_dim)
        self.layers['dec_4'] = LSTMDecodeLayer(self.dec_dim)

    def _get_init_state(self, batch_size):
        return [tensor.alloc(dtype_cast(0.), batch_size, self.dec_dim) for i in range(9)]

    def _step(self, mask, y_tm1, c1_tm1, c2_tm1, c3_tm1, c4_tm1, h1_tm1, h2_tm1, h3_tm1, h4_tm1,context_vector):

        emb_y_tm1 = y_tm1
        if self.use_dropout:
            h1, c1 = self.layers['dec_1'].forward(emb_y_tm1, emb_y_tm1, mask, h1_tm1, c1_tm1, context_vector)
            h2, c2 = self.layers['dec_2'].forward(dropout(h1), emb_y_tm1, mask, h2_tm1, c2_tm1, context_vector)
            h3, c3 = self.layers['dec_3'].forward(dropout(h2), emb_y_tm1, mask, h3_tm1, c3_tm1, context_vector)
            h4, c4 = self.layers['dec_4'].forward(dropout(h3), emb_y_tm1, mask, h4_tm1, c4_tm1, context_vector)

        else:
            h1, c1 = self.layers['dec_1'].forward(emb_y_tm1, emb_y_tm1, mask, h1_tm1, c1_tm1, context_vector)
            h2, c2 = self.layers['dec_2'].forward(h1, emb_y_tm1, mask, h2_tm1, c2_tm1, context_vector)
            h3, c3 = self.layers['dec_3'].forward(h2, emb_y_tm1, mask, h3_tm1, c3_tm1, context_vector)
            h4, c4 = self.layers['dec_4'].forward(h3, emb_y_tm1, mask, h4_tm1, c4_tm1, context_vector)

        y = h4
        return y, c1, c2, c3, c4, h1, h2, h3, h4

    def decode(self, context_vector, target_seq_mask):
        batch_size = context_vector.shape[0]
        init_state = self._get_init_state(batch_size)
        rval, _ = theano.scan(
            name          = 'Decoder',
            fn            = self._step,
            sequences     = target_seq_mask,
            outputs_info  = init_state,
            non_sequences = context_vector,
        )
        return rval[1]


class WordDecoder(Decoder):
    def __init__(self, dec_dim, vocab_size, word_emb_layer, use_dropout):
        Decoder.__init__(self, dec_dim, use_dropout)
        self.vocab_size = vocab_size
        self.layers['output_layer'] = OutputLayer(self.dec_dim, self.vocab_size)
        self.word_emb_layer = word_emb_layer

    def _step(self, mask, y_tm1, c1_tm1, c2_tm1, c3_tm1, c4_tm1, h1_tm1, h2_tm1, h3_tm1, h4_tm1, context_vector):

        emb_y_tm1 = self.word_emb_layer.forward(y_tm1)
        if self.use_dropout:
            h1, c1 = self.layers['dec_1'].forward(emb_y_tm1, emb_y_tm1, mask, h1_tm1, c1_tm1, context_vector)
            h2, c2 = self.layers['dec_2'].forward(dropout(h1), emb_y_tm1, mask, h2_tm1, c2_tm1, context_vector)
            h3, c3 = self.layers['dec_3'].forward(dropout(h2), emb_y_tm1, mask, h3_tm1, c3_tm1, context_vector)
            h4, c4 = self.layers['dec_4'].forward(dropout(h3), emb_y_tm1, mask, h4_tm1, c4_tm1, context_vector)
            y, y_idx = self.layers['output_layer'].forward(y_tm1, dropout(h4), context_vector)

        else:
            h1, c1 = self.layers['dec_1'].forward(emb_y_tm1, emb_y_tm1, mask, h1_tm1, c1_tm1, context_vector)
            h2, c2 = self.layers['dec_2'].forward(h1, emb_y_tm1, mask, h2_tm1, c2_tm1, context_vector)
            h3, c3 = self.layers['dec_3'].forward(h2, emb_y_tm1, mask, h3_tm1, c3_tm1, context_vector)
            h4, c4 = self.layers['dec_4'].forward(h3, emb_y_tm1, mask, h4_tm1, c4_tm1, context_vector)
            y, y_idx = self.layers['output_layer'].forward(y_tm1, h4, context_vector)

        return y, c1, c2, c3, c4, h1, h2, h3, h4


class SentDecoder(Decoder):
    def __init__(self, dec_dim, use_dropout):
        Decoder.__init__(self, dec_dim, use_dropout)
