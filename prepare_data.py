#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import json

current_path = os.getcwd()
vocabs_path = r'C:\Users\Azusa\Desktop\ACL2012_wordVectorsTextFile\vocab.txt'
word_table_path = r'C:\Users\Azusa\Desktop\ACL2012_wordVectorsTextFile\wordVectors.txt'
vocabs_save_path = current_path + r'/vocabs.json'
word_table_save_path = current_path + r'/word_table.json'
vocab_size = 100000
emb_dim = 50

def get_vocabs():
    with open(vocabs_path, 'r') as file:
        s = file.read()

    number = range(1, vocab_size)
    words = s.split('\n')
    words = words[:vocab_size-1]
    assert words[0] == 'UUUNKKK' # represent unknow token
    print('Get %s words and UNK token' % len(words))
    vocabs = dict(zip(words, number))

    with open(vocabs_save_path, 'w') as file:
        file.write(json.dumps(vocabs))


def get_word_table():
    with open(word_table_path, 'r') as file:
        s = file.readlines()

    x = [[0.0] for i in range(emb_dim)]
    for i, line in enumerate(s):
        cc = line.strip('\n').rstrip().split(' ')
        val = [float(k) for k in cc]
        x.append(val)
        if i == vocab_size - 2:
            break

    with open(word_table_save_path, 'w') as file:
        file.write(json.dumps(x))


def test():
    with open(vocabs_save_path, 'r') as file:
        vocabs = json.loads(file.read())
    with open(word_table_save_path, 'r') as file:
        word_table = json.loads(file.read())

    for val in vocabs.values():
        if val == 0:
            print('Format Error')
            return -1

    if len(vocabs) != vocab_size - 1:
        print('Format Error')
        return -1

    if vocabs['UUUNKKK'] != 1:
        print('Format Error')
        return -1

    print('vocab file format correct')

    if len(word_table) != vocab_size:
        print(len(word_table))
        print('Format Error')
        return -1

    if len(word_table[1]) != emb_dim:
        print(len(word_table[1]))
        print('Format Error')
        return -1

    print(max(vocabs.values()))

    print('vocab file format correct')
    print(word_table[0])
    print('\n')
    print(word_table[1])


if __name__ == '__main__':
    #get_vocabs()
    #get_word_table()
    test()
