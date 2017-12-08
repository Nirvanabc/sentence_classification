import struct
from nltk.tokenize import sent_tokenize
import re
import numpy as np
from numpy.random import normal
from random import shuffle
import pickle

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


def corpora2vec(corpora, vec_size):
    result = []
    for sent in corpora:
        curr = []
        for word in sent:
            curr.append(normalize(rand(vec_size)))
        result.append(curr)
    return result


def padd_sent_to_size(sent, vec_size, sent_size):
    res_sent = np.zeros([sent_size, vec_size])
    if len(sent) <= sent_size:
        res_sent[sent_size - len(sent):] = sent
    else: res_sent = sent[:sent_size]
    return res_sent


def padd_and_label_corpora(corpora, vec_size, sent_size, label):
    res_corpora = []
    for sent in corpora:
        res_corpora.append([padd_sent_to_size(sent,         \
                                              vec_size,     \
                                              sent_size)] + \
                           [label])
    return res_corpora


def del_empty(corpora):
    return list(filter(lambda x: x!=[], corpora))


def prepare_corpora(file_name, vec_size, \
                    sent_size, label):
    with open(file_name, 'rb') as f:
        corpora = pickle.load(f)
        f.close()
    corpora = del_empty(corpora)
    corpora = corpora[:2]
    vec_dictionary = corpora2vec(corpora, vec_size)
    vec_dictionary = padd_and_label_corpora(vec_dictionary, \
                                            vec_size,       \
                                            sent_size,      \
                                            label)
    shuffle(vec_dictionary)
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

    vec_dictionary_bad = prepare_corpora(bad,     \
                                         vec_size, sent_size, \
                                         label_bad)
    vec_dictionary_good = prepare_corpora(good,    \
                                          vec_size, sent_size, \
                                          label_good)
    
    return vec_dictionary_bad, vec_dictionary_good, vec_size
