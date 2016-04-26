#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  ------ Sentence Autoencoder ------
  Created by Xiaolin Wan, 4.24.2016
  Last updated in 4.26.2016

  Build a deeply sentence autoencoder with Bi-LSTM.
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

class SAE:
    def __init__(self, sae_struct_params, sae_training_params):
        # Set the random number generators' seeds for consistency
        self.SEED = 888
        self.trng = RandomStreams(self.SEED)
        numpy.random.seed(self.SEED)

        self.nn_weights = {}
        self.struct_params   = sae_struct_params
        self.training_params = sae_training_params
        self.__init_nn_weights()


    def __numpy_floatX(self, data):
        return numpy.asarray(data, dtype=config.floatX)


    def __get_ortho_weights(self, dim):
        tmp = numpy.random.randn(dim, dim)  # 方阵
        ortho_weight = numpy.linalg.svd(tmp)
        concat_ortho_weight = numpy.concatenate([ortho_weight, ortho_weight, ortho_weight, ortho_weight], axis=1)
        return concat_ortho_weight


    def __numpy2theano(self):
        pass # TODO
    
    
    def __slice(self, _x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]


    def __init_nn_weights(self):
        """
        In this model, Neural network weights includes:
        Encoder:
            enc_W[i], enc_U[i], enc_V[i], enc_b[i]

        Decoder:
            dec_W[i], dec_U[i], dec_V[i], dec_b[i]

        Feature Extraction:
            mlp_W[i], mlp_b[i]
        """
        assert self.nn_weights is None

        enc_ortho_weight = self.__get_ortho_weights(self.struct_params['enc_hidden_dim'])
        dec_ortho_weight = self.__get_ortho_weights(self.struct_params['dec_hidden_dim'])

        for i in range(0, self.struct_params['enc_n_layers']):
            self.nn_weights['enc_W'].append(enc_ortho_weight)
            self.nn_weights['enc_U'].append(enc_ortho_weight)
            self.nn_weights['enc_V'].append(enc_ortho_weight)
            self.nn_weights['enc_b'].append(numpy.zeros((4 * self.struct_params['enc_hidden_dim'],)))

        for i in range(0, self.struct_params['enc_n_layers']):
            self.nn_weights['dec_W'].append(dec_ortho_weight)
            self.nn_weights['dec_U'].append(dec_ortho_weight)
            self.nn_weights['dec_V'].append(dec_ortho_weight)
            self.nn_weights['dec_b'].append(numpy.zeros((4 * self.struct_params['enc_hidden_dim'],)))




    def __get_dropout_layer(self, state_before, noise):
        censored = state_before * self.trng.binomial(state_before.shape,p=0.5, n=1, dtype=state_before.dtype)
        output = tensor.switch(cond = noise,
                               ift  = censored,
                               iff  = state_before * 0.5)
        return output

    def __get_2maxout_layer(self, x):
        # TODO:
        return x


    def encoder_lstm_unit(self, x_, m_, h_, c_, n_layer): # TODO: 根据scan()的行为调整参数顺序
        preact = tensor.dot(h_, self.nn_weights['enc_U'][n_layer])
        preact += x_

        i = tensor.nnet.sigmoid(self.__slice(preact, 0, self.struct_params['hidden_units_dim']))
        f = tensor.nnet.sigmoid(self.__slice(preact, 1, self.struct_params['hidden_units_dim']))
        o = tensor.nnet.sigmoid(self.__slice(preact, 2, self.struct_params['hidden_units_dim']))
        c = tensor.tanh(self.__slice(preact, 3, self.struct_params['hidden_units_dim']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
    
    
    def decoder_lstm_unit(self, x_, m_, a_, h_, c_, n_layer):
        preact = (tensor.dot(h_, self.nn_weights['dec_U'][n_layer]) +
                  tensor.dot(a_, self.nn_weights['dec_V'][n_layer]))
        preact += x_

        i = tensor.nnet.sigmoid(self.__slice(preact, 0, self.struct_params['hidden_units_dim']))
        f = tensor.nnet.sigmoid(self.__slice(preact, 1, self.struct_params['hidden_units_dim']))
        o = tensor.nnet.sigmoid(self.__slice(preact, 2, self.struct_params['hidden_units_dim']))
        c = tensor.tanh(self.__slice(preact, 3, self.struct_params['hidden_units_dim']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c


    def get_encoder_layer(self, input_seq, mask, below_layer, n_layer):
        nsteps = input_seq.shape[0]
        if input_seq.ndim == 3:
            n_samples = input_seq.shape[1]
        else:
            n_samples = 1

        init_val = [tensor.alloc(self.__numpy_floatX(0.), n_samples, self.struct_params['hidden_units_dim']),
                    tensor.alloc(self.__numpy_floatX(0.), n_samples, self.struct_params['hidden_units_dim'])]

        state = (tensor.dot(input_seq, self.nn_weights['enc_W'][n_layer]) +
                 tensor.dot(below_layer, self.nn_weights['enc_V'][n_layer]) +
                 self.nn_weights['enc_b'][n_layer])

        layer_output, update = theano.scan(name          = '',
                                           fn            = self.encoder_lstm_unit,
                                           sequences     = [state, mask],
                                           outputs_info  = init_val,
                                           non_sequences = n_layer,
                                           n_steps       = nsteps)
        return layer_output


    def get_decoder_layer(self, input_seq, mask, below_layer, context_vector, n_layer):
        nsteps = input_seq.shape[0]
        if input_seq.ndim == 3:
            n_samples = input_seq.shape[1]
        else:
            n_samples = 1

        init_val = [tensor.alloc(self.__numpy_floatX(0.), n_samples, self.struct_params['hidden_units_dim']),
                    tensor.alloc(self.__numpy_floatX(0.), n_samples, self.struct_params['hidden_units_dim'])]

        state = (tensor.dot(input_seq, self.nn_weights['dec_W'][n_layer]) +
                 tensor.dot(context_vector, self.nn_weights['dec_C'][n_layer]) +
                 self.nn_weights['enc_b'][n_layer])

        layer_output, update = theano.scan(name          = '',
                                           fn            = self.decoder_lstm_unit,
                                           sequences     = [state, mask, below_layer],
                                           outputs_info  = init_val,
                                           non_sequences = n_layer,
                                           n_steps       = nsteps)
        return layer_output


    def get_encoder(self, x, x_mask):
        """
        Encoder:
        We build a four layers encoder without dropout, dropout layers will be planted
        """
        encoder_layers = self.struct_params['encoder_layers']

        h1 = self.get_encoder_layer(x, x_mask, x, 0)
        h2 = self.get_encoder_layer(x, x_mask, h1, 1)
        h3 = self.get_encoder_layer(x, x_mask, h2, 2)
        h4 = self.get_encoder_layer(x, x_mask, h3, 3)

        return h4[-1]


    def get_decoder(self, y, y_mask, context_vector):
        """
        decoder:
        We build a four layers decoder without dropout, dropout layers will be planted
        """
        decoder_layers = self.struct_params['decoder_layers']

        h1 = self.get_decoder_layer(y, y_mask, y, context_vector, 0) # TODO:
        h2 = self.get_decoder_layer(y, y_mask, h1, context_vector, 1)
        h3 = self.get_decoder_layer(y, y_mask, h2, context_vector, 2)
        h4 = self.get_decoder_layer(y, y_mask, h3, context_vector, 3)

        return h4


    def build_sae_model(self):
        """

        """
        # Used for dropout.
        use_noise = theano.shared(self.__numpy_floatX(0.))

        x = tensor.matrix('x', dtype='') # TODO
        x_mask = tensor.matrix('x_mask', dtype='')
        y = tensor.matrix('y', dtype='')
        y_mask = tensor.matrix('y_mask', dtype='')
        forward_encoder, backward_encoder = self.get_encoder

        n_timesteps = x.shape[0]
        n_samples   = x.shape[1]

        # ------ Encoder ------
        rx = reversed(x)
        rx_mask = reversed(x_mask)

        h_fenc = forward_encoder(x, x_mask)
        h_benc = backward_encoder(rx, rx_mask)

        h_enc = tensor.concatenate([h_fenc, h_benc], axis=1)

        # ------ Feature Extraction ------
        # We use MLP (Multi Layer Perceptron) to get high-level representation.
        # The MLP is 2 layers
        # In the future, the CNN (Convolutional Neural Network) may be added in
        h1_mlp = tensor.nnet.relu(tensor.dot(h_enc, self.nn_weights['mlp_W'][0] + self.nn_weights['mlp_b'][0]), alpha=0)
        h2_mlp = tensor.nnet.relu(tensor.dot(h1_mlp, self.nn_weights['mlp_W'][1] + self.nn_weights['mlp_b'][1]), alpha=0)

        # ------ Decoder ------
        decoder = self.get_decoder

        h_dec = decoder() # TODO:

        # ------ 2-Maxout Layer ------
        assert x.shape() % 2 == 0
        s_2maxout = self.__get_2maxout_layer(h_dec)

        # ------ Softmax Layer ------
        prediction = tensor.nnet.softmax(tensor.dot(s_2maxout, self.nn_weights['softmax_W']) +
                                         self.nn_weights['softmax_b'])

        # ------ Output ------
        f_pred_prob = theano.function([x, x_mask], prediction, name='f_pred_prob')
        f_pred = theano.function([x, x_mask], prediction.argmax(axis=1), name='f_pred')

        off = 1e-8
        if prediction.dtype == 'float16':
            off = 1e-6

        cost = -tensor.log(prediction[tensor.arange(n_samples), y] + off).mean()

        return use_noise, x, x_mask, y, f_pred_prob, f_pred, cost
































