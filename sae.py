#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  ------Sentence Autoencoder------
  Created by Xiaolin Wan, 4.24.2016

  Build a deep sentence autoencoder with Bi-LSTM.
  We adopt a Bi-LSTM structure with four layer for encoding and four layer for decoding,
  a deep structure is planted between encoder and decoder to extract high-level sentence representation.

  Draft* Do NOT cite.
"""

import numpy
import theano
import theano.tensor as tensor
from theano import config

class SAE:
    def __init__(self, sae_structure_params, sae_training_params):
        SEED = 888
        numpy.random.seed(SEED)

        self.struct_params   = sae_structure_params
        self.training_params = sae_training_params

        self.nn_weights = self.__init_nn_weights() # TODO: initialize with init_sae_params


    def __numpy_floatX(self, data):
        return numpy.asarray(data, dtype=config.floatX)

    def init_params(self):
        pass


    def __init_nn_weights(self):
        hidden_units_dim = self.struct_params['hidden_units_dim']

        def _get_ortho_weight(hidden_units_dim):
            tmp = numpy.random.randn(hidden_units_dim, hidden_units_dim)
            ortho_weight = numpy.linalg.svd(tmp)
            pp = numpy.concatenate([ortho_weight,ortho_weight,ortho_weight,ortho_weight], axis=1)
            return pp

        for ww in self.nn_weights['enc_W']:
            ww = _get_ortho_weight(hidden_units_dim)
        for uu in self.nn_weights['enc_U']:
            uu = _get_ortho_weight(hidden_units_dim)
        for bb in self.nn_weights['enc_b']:
            bb = numpy.zeros((4 * self.struct_params[''],))
        for ww in self.nn_weights['dec_W']:
            ww = _get_ortho_weight(hidden_units_dim)
        for uu in self.nn_weights['dec_U']:
            uu = _get_ortho_weight(hidden_units_dim)
        for bb in self.nn_weights['dec_b']:
            bb = numpy.zeros((4 * self.struct_params[''],))
        # TODO: This part of the code will be rewritten
        return 0


    def dropout_layer(self, layer_output):
        return layer_output


    def __lstm_unit(self, x_, m_, h_, c_, lstm_tag):

        layer_num = 0

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        preact = tensor.dot(h_, self.nn_weights[lstm_tag][layer_num])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, self.struct_params['hidden_units_dim']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, self.struct_params['hidden_units_dim']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, self.struct_params['hidden_units_dim']))
        c = tensor.tanh(_slice(preact, 3, self.struct_params['hidden_units_dim']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c


    def get_encoder_last_layer(self, x_seq, mask, iter_direction):
        nsteps = x_seq.shape[0]
        if x_seq.ndim == 3:
            n_samples = x_seq.shape[1]
        else:
            n_samples = 1

        layer_num = self.struct_params['encoder_n_layers']
        init_val = [tensor.alloc(self.__numpy_floatX(0.), n_samples, self.struct_params['hidden_units_dim']),
                    tensor.alloc(self.__numpy_floatX(0.), n_samples, self.struct_params['hidden_units_dim'])]

        y = (tensor.dot(x_seq, self.nn_weights['enc_W'][layer_num]) +
             self.nn_weights['enc_b'][layer_num])
        rval, update = theano.scan(name          = '',
                                   fn            = self.__lstm_unit,
                                   sequences     = [x_seq, mask],
                                   outputs_info  = init_val,
                                   non_sequences = 'enc_U', # we only compute U*x in LSTM
                                   n_steps       = nsteps)
        return rval[-1]


    def get_decoder_first_layer(self, y_seq, mask, c):

        nsteps = y_seq.shape[0]
        if y_seq.ndim == 3:
            n_samples = y_seq.shape[1]
        else:
            n_samples = 1

        layer_num = self.struct_params['decoder_n_layers']
        assert layer_num == 1 #todo

        h_0 = tensor.tanh(tensor.dot(self.nn_weights['V'], c))

        y = (tensor.dot(y_seq, self.nn_weights['dec_W'][layer_num]) +
             tensor.dot(c, self.nn_weights['dec_C'][layer_num]) +
             self.nn_weights['dec_b'][layer_num])
        rval, update = theano.scan(name          = '',
                                   fn            = self.__lstm_unit,
                                   sequences     = [y_seq, mask],
                                   outputs_info  = h_0,
                                   non_sequences = 'dec_U',
                                   n_steps       = nsteps)



    def get_decoder_layer(self, h_seq, mask):

        nsteps = h_seq.shape[0]
        if h_seq.ndim == 3:
            n_samples = h_seq.shape[1]
        else:
            n_samples = 1

        layer_num = self.struct_params['decoder_n_layers']
        assert layer_num == 1 #todo

        h_0 = tensor.tanh(tensor.dot(self.nn_weights['V'])) # todo

        y = (tensor.dot(h_seq, self.nn_weights['dec_W'][layer_num]) +
             self.nn_weights['dec_b'][layer_num])
        rval, update = theano.scan(name          = '',
                                   fn            = self.__lstm_unit,
                                   sequences     = [h_seq, mask],
                                   outputs_info  = h_0,
                                   non_sequences = 'dec_U',
                                   n_steps       = nsteps)


    # TODO: Before concatenate all layers, the MLP layer must be used.
    # TODO: Remember: different Xcoder forms.
    def build_sae_model(self):
        # TODO：prehandle data
        encoder_n_layers = self.struct_params['']
        decoder_n_layers = self.struct_params['']

        #TODO Build sentence encoder, using get_encoder_layer()

        #TODO Build sentence decoder, using get_decoder_layer()


    def sgd(self):
        """
        ------Stochastic Gradient Descent------
        """
        pass


    def save_model(self):
        pass


    def load_model(self):
        pass
















