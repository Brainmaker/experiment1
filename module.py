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
        self.submodule = OrderedDict().fromkeys(submodule_name)

    def save(self, filepath):
        container = OrderedDict().fromkeys(self.submodule.keys())
        for name in self.submodule.keys():
            container[name] = self.submodule[name].layers2json()
        module_json = json.dumps(container)
        with open(filepath, 'w') as f:
            f.write(module_json)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            module_json = f.read()
        container = json.loads(module_json)
        assert container.keys() == self.submodule.keys()
        for name in container.keys():
            self.submodule[name].json2layers(container[name])

    def get_params(self):
        module_params = []
        for kk in self.submodule.keys():
            module_params.extend(self.submodule[kk].get_params())
        return module_params

# 下面不加注释把我自己都绕晕了


class SAE(Module):
    """
        SAE的输入:
        x: 一个包含mini batch中所有句子的三维数组。每一列代表一个句子，每一行代表一个词。即一个timestep。
           其大小为 max_timestep * batch_size * vocab_size

        mask: 一个包含mini batch中所有句子mask的二维数组。每一列代表一个句子mask，每一行代表一个词mask。
              其大小为 max_timestep * batch_size
    """
    def __init__(self, vocab_size, enc_dim, dec_dim, use_dropout):
        Module.__init__(self, ['word_enc', 'word_dec'])
        self.vocab_size = vocab_size
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.use_dropout = use_dropout

        self.submodule['word_enc'] = WordEncoder(self.enc_dim, self.vocab_size, use_dropout=use_dropout)
        wemb_layer = self.submodule['word_enc'].layers['word_embedding']
        self.submodule['word_dec'] = WordDecoder(self.dec_dim, self.vocab_size, wemb_layer, use_dropout=use_dropout)

    def get_context_vector(self, x, mask):
        s_emb = self.submodule['word_enc'].word_encode(x, mask)
        context_vector = s_emb.T[0:3].T  # TODO: test 这里一定要注意！！！！！！加入更多的层后删掉！！
        return context_vector

    def decode(self, context_vector, mask):
        pred_seq, prob_pred_seq = self.submodule['word_dec'].decode(context_vector, mask)
        return pred_seq, prob_pred_seq

    def forward(self, x, mask):
        context_vector = self.get_context_vector(x, mask)
        pred_seq, prob_pred_seq = self.decode(context_vector, mask)
        return pred_seq, prob_pred_seq


class DAE(Module):
    """
        DAE的输入:
        x: 一个包含mini batch中所有文档的四维数组。对于前两个维而言，每一行代表一个timestep（即一个句子），每一列代表一个sample。
           其大小为 max_doc_length * batch_size * max_sent_length * vocab_size

        sent_mask: 大小为max_doc_length * batch_size * max_sent_length

        doc_mask: 大小为max_doc_length * batch_size
    """
    def __init__(self, enc_dim, dec_dim, sae, use_dropout):
        Module.__init__(self, ['sent_enc', 'sent_dec'])
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.use_dropout = use_dropout
        self.sae = sae  # 在使用DAE前，必须先训练SAE
        self.submodule['sent_enc'] = SentEncoder(self.enc_dim, use_dropout=use_dropout)
        self.submodule['sent_dec'] = SentDecoder(self.dec_dim, use_dropout=use_dropout)

    def get_context_vector(self, x, sents_mask, doc_mask, max_doc_length):

        # x[i]大小为 batch_size * max_sent_length * vocab_size
        # sent_mask[i]大小为 batch_size * max_sent_length
        # 因此需要交换维度batch_size与max_sent_length

        s_cv_list = [self.sae.get_context_vector(x[i].dimshuffle(1, 0, 2), sents_mask[i].dimshuffle(1, 0))
                     for i in range(max_doc_length)]  # todo

        # 对每个i, sae返回的每个context vector都是大小为 batchsize * enc_dim 的二维数组
        # 接下来，用stack()将其合并。得到的三维数组大小为 max_doc_length * batchsize * enc_dim

        s_cv = tensor.stack(s_cv_list, axis=2).dimshuffle(2, 0, 1)
        context_vector = self.submodule['sent_enc'].encode(s_cv, doc_mask)
        return context_vector

    def decode(self, context_vector, sents_mask, doc_mask, max_doc_length):
        s_cv, _ = self.submodule['sent_dec'].decode(context_vector, doc_mask)

        # context_vector大小: batch_size * enc_dim
        # s_cv大小: max_doc_length * batchsize * enc_dim
        # sent_mask大小: max_doc_length * batch_size * max_sent_length

        sents_mask = sents_mask.dimshuffle(0, 2, 1)
        # prob_pred_seq_list = [self.sae.decode(s_cv[0], sents_mask[0]), self.sae.decode(s_cv[1], sents_mask[1])]
        prob_pred_seq, _ = theano.scan(
            fn        = self.sae.decode,
            sequences = [s_cv, sents_mask],
            n_steps   = max_doc_length
        )

        # self.submodule['word_dec'].decode(s_cv[i], sents_mask[i])得到的内容为
        #     pred_seq: max_sents_length * batchsize * vocab_size (one-hot vector)
        #     pred_prob_seq: max_sents_length * batchsize * vocab_size (real value vector)
        # 这里只取pred_prob_seq为列表内容，列表长度为max_doc_length

        #prob_pred_seq = tensor.stack(prob_pred_seq_list[1], axis=3)

        # return s_cv.shape, sents_mask.dimshuffle(0, 2, 1).shape
        return prob_pred_seq[0], prob_pred_seq[1]

    def forward(self, x, sent_mask, doc_mask, max_doc_length):
        """

        """
        context_vector = self.get_context_vector(x, sent_mask, doc_mask, max_doc_length).T[0:3].T
        # TODO: test 这里一定要注意！！！！！！加入更多的层后删掉！！

        # decode()返回的张量pred_seq, prob_pred_seq同尺寸，其大小为：
        #   max_doc_length * max_sents_length * batch_size * vocab_size
        # 使用dimshuffle交换axis后返回与target_seq对应的tensor
        # 将来是否使用dimshuffle取决于train()的输入数据格式
        pred_seq, prob_pred_seq = self.decode(context_vector, sent_mask, doc_mask, max_doc_length)
        return pred_seq.dimshuffle(0, 2, 1, 3).shape, prob_pred_seq.dimshuffle(0, 2, 1, 3).shape
