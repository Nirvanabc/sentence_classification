import struct
from nltk.tokenize import sent_tokenize
import re
import numpy as np
from numpy.random import normal
from random import shuffle
import pickle

def read_word_and_its_vec(opened_file, vec_len):
    try:
        char = opened_file.read(1)
        word = b''
        while char != b' ':
            word += char
            char = opened_file.read(1)
        vec = np.empty(vec_len)
        for i in range(vec_len):
            num = struct.unpack('f', opened_file.read(4))
            vec[i] = num[0]
        char = opened_file.read(1)
        word = word.decode()
    finally:
        return word, vec


def get_dict(dict_file):
    '''
    returns a dict of all words, words 2-3 min for eng
    and 17-19 min for rus
    '''
    my_dict = open(dict_file, 'rb')
    line = my_dict.readline()
    line = line.split()
    row = int(line[0])
    col = int(line[1])
    result_dict = {}
    for _ in range(row):
        word, vec = read_word_and_its_vec(my_dict, col)
        result_dict[word] = vec
    return result_dict, col


def normalize(vec):
    total = sum(vec)
    return list(map(lambda x: x/total, vec))


def word2vec(word, dictionary, vec_size):
    try:
        result = dictionary[word]
        return result
    except KeyError:
        new_vec = normalize(normal(size = vec_size))
        result = dictionary[word] = new_vec
        return result


def corpora2vec(corpora, dictionary, vec_size):
    result = []
    for sent in corpora:
        curr = []
        for word in sent:
            curr.append(word2vec(word, dictionary, vec_size))
        result.append(curr)
    return result


def padd_sent_to_size(sent, vec_size, sent_size):
    res_sent = np.zeros([sent_size, vec_size])
    if len(sent) <= sent_size:
        res_sent[sent_size - len(sent):] = sent
    else: res_sent = sent[:sent_size]
    return res_sent


def padd_and_label_corpora(corpora, vec_size, sent_size):
    res_corpora = []
    for sent in corpora:
        res_corpora.append(padd_sent_to_size(sent,         \
                                              vec_size,     \
                                              sent_size))
    return res_corpora


def del_empty(corpora):
    return list(filter(lambda x: x!=[], corpora))


def prepare_corpora(corpora, dictionary, vec_size, \
                    sent_size):
    '''
    takes a batch and prepare it
    '''
    vec_dictionary = corpora2vec(corpora, dictionary,  vec_size)
    vec_dictionary = padd_and_label_corpora(vec_dictionary, \
                                            vec_size,       \
                                            sent_size)
    return vec_dictionary    


def rand(vec_size):
    return normalize(normal(size = vec_size))


def store_data(data, file_to_store):
    f = open(file_to_store, 'wb')
    pickle.dump(data, f)
    f.close()

    
def my_dictionary():
    vec_size = 300
    sent_size = 16
    bad = 'ready_bad'
    good = 'ready_good'
    label_bad = 0
    label_good = 1
    ru_dict_source = 'softlink_ru'
    en_dict_source = 'softlink_en'
    dictionary, vec_size = get_dict(ru_dict_source)
    vec_dictionary_bad = prepare_corpora(bad, dictionary,     \
                                         vec_size, sent_size, \
                                         label_bad)
    vec_dictionary_good = prepare_corpora(good, dictionary,    \
                                          vec_size, sent_size, \
                                          label_good)
    data = vec_dictionary_good + vec_dictionary_bad
    shuffle(data)
    return data, vec_size


def next_batch(corpora, n):
    batch = []
    i = 0
    for _ in range(n):
        sent = corpora.readline()
        if len(sent) == 0:
            return 0
        sent = sent.split()
        label = int(sent[-1])
        sent = [sent[:-1], [1-label, label]]
        batch.append(sent)
    batch = prepare_corpora(batch, dictionary, \
                            vec_size, sent_size)
    return batch


def my_dictionary2(batch_size):
    # vec_size = 300
    sent_size = 16
    ru_dict_source = 'softlink_ru'
    en_dict_source = 'softlink_en'
    dictionary, vec_size = get_dict(ru_dict_source)
