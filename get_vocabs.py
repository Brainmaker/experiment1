#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path
import json
from collections import defaultdict, Counter

MOST_COMMON = 120000

savepath = r'D:\My Documents\My Project\experiment1\finished\test_vocabs.json'
dirpath = 'D:\\My Documents\\My Project\\experiment1\\finished\\test'
#dirpath = 'D:\\Corpus\\1-billion-word-language-modeling-benchmark-r13output\\1-billion-word-language-modeling-benchmark-r13output\\training-monolingual.tokenized.shuffled'
#savepath = 'D:\\My Documents\\My Project\\experiment1\\finished\\a.json'
#dirpath = 'D:\\My Documents\\My Project\\experiment1\\finished\\test'

def get_file_vocabs(file):
    file_vocabs = Counter()
    for sent in file.readlines():
        voc = Counter()
        for word in sent.split():
            voc[word] += 1
        file_vocabs.update(voc)
    return file_vocabs

def get_vocab(dirpath):
    vocabs = {}
    cvocabs = Counter()
    for filename in os.listdir(dirpath):
        with open(dirpath + '\\' + filename, 'r', encoding='utf-8') as file:
            file_vocabs = get_file_vocabs(file)
            cvocabs.update(file_vocabs)
            print('Step 1: Process file', filename)

    n = len(cvocabs)
    if n >= MOST_COMMON: n = MOST_COMMON
    cvocabs = dict(cvocabs.most_common(n))

    print('Step 2...')
    for i, kk in enumerate(cvocabs.keys()):
        vocabs[kk] = i + 1

    return vocabs

if __name__ == '__main__':
    vocabs = get_vocab(dirpath)
    print('Saving...')
    with open(savepath, 'w') as file:
       file.write(json.dumps(vocabs))
