#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from collections import OrderedDict

import theano
import theano.tensor as tensor

from submodule import \
    WordEncoder, SentEncoder, WordDecoder, SentDecoder

class Module(object):
    def __init__(self, submodule_name):
        self.submodule = OrderedDict().fromkeys([submodule_name])

    def save(self, filepath):
        container = OrderedDict().fromkeys(self.submodule.keys())
        for name in self.submodule.keys():
            container[name] = self.submodule[name]
        module_json = json.dumps(container)
        with open(filepath, 'w') as f:
            f.write(module_json)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            module_json = f.read()
        assert module_json.keys() == self.submodule.keys
        for name in module_json.keys():
            self.submodule[name] = module_json[name]

    def get_params(self):
        pass


class SAE(Module):
    def __init__(self, vocab_size, enc_dim, dec_dim, use_dropout):
        Module.__init__(self, ['word_enc','word_dec'])
        self.vocab_size = vocab_size
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.use_dropout = use_dropout

        self.submodule['word_enc'] = WordEncoder(self.enc_dim, self.vocab_size, use_dropout=use_dropout)
        wemb_layer = self.submodule['word_enc'].layers['word_embedding']
        self.submodule['word_dec'] = WordDecoder(self.dec_dim, self.vocab_size, use_dropout=use_dropout)

    def get_context_vector(self, x, mask):
        s_emb = self.submodule['word_enc'].word_encode(x, mask)
        context_vector = s_emb
        return context_vector

    def decode(self, context_vector, mask):
        pred_seq = self.submodule['word_dec'].decode(context_vector, mask)
        return pred_seq

    def forward(self, x, mask):
        context_vector = self.get_context_vector(x, mask)
        pred_seq = self.decode(context_vector, mask)
        return pred_seq


class DAE(Module):
    def __init__(self, enc_dim, dec_dim, sae, use_dropout):
        Module.__init__(self, ['sent_enc','sent_dec'])
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.use_dropout = use_dropout
        self.sae = sae # 在使用DAE前，必须先训练SAE
        self.submodule['sent_enc'] = SentEncoder(self.enc_dim, use_dropout=use_dropout)
        self.submodule['sent_dec'] = SentDecoder(self.dec_dim, use_dropout=use_dropout)

    def get_context_vector(self, x, sents_mask, doc_mask):
        batch_size = x.shape[0]
        s_cv = [self.sae.get_context_vector(x[i], sents_mask[i]) for i in range(batch_size)]
        # TODO:scv可能需要一些处理, 需要测试
        context_vector = self.submodule['sent_enc'].encode(s_cv, doc_mask)
        return context_vector

    def decode(self, context_vector, sents_mask, doc_mask):
        s_cv = self.submodule['sent_dec'].decode(context_vector, doc_mask)
        # TODO:scv可能需要一些处理, 需要测试
        pred_seq = 0 #TODO
        return pred_seq

    def forward(self, x, sent_mask, doc_mask):
        context_vector = self.get_context_vector(x, sent_mask, doc_mask)
        pred_seq = self.decode(context_vector, sent_mask, doc_mask)
        return pred_seq



































