# works 3 min for eng, 18 for rus (on leplop)
# and 5 min on rus on Dima's computer (use ssh)

import struct
from nltk.tokenize import sent_tokenize
import re
import numpy as np
from numpy.random import normal
import pickle
# lemmatization if needed
# from pymystem3 import Mystem
# m = Mystem()    

def normalize(vec):
    total = sum(vec)
    return list(map(lambda x: x/total, vec))
    

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


def prepare_corpora(file_name):
    '''
    returns such data: [['трезво', 'мыслящие'], ['я']]
    for a given corpora
    '''
    with open(file_name, 'rb') as f:
        corpora = pickle.load(f)
    result = []
    for review in corpora:
        for i in sent_tokenize(review):
            text = re.sub('\!caret_return\!', '', i).lower()
            text = re.sub('\W', ' ', text)
            text = re.sub(' +', ' ', text)
            # text = m.lemmatize(text)
            # text = list(filter(lambda x: x != ' \n' and \
            #                              x != ' ', text))
            result.append(text.split())
    return result


def store_data(data, file_to_store):
    f = open(file_to_store, 'wb')
    pickle.dump(data, f)
    f.close()


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


def padd_corpora_to_size(corpora, vec_size, sent_size):
    res_corpora = []
    for sent in corpora:
        res_corpora.append(padd_sent_to_size(sent, \
                                             vec_size, \
                                             sent_size))
    return res_corpora


def prepare_corpora(file_name, dictionary, vec_size, \
                    sent_size):
    with open(file_name, 'rb') as f:
        corpora = pickle.load(f)
        f.close()
    vec_dictionary = corpora2vec(corpora, dictionary, \
                                 vec_size)
    vec_dictionary = padd_corpora_to_size(vec_dictionary, \
                                          vec_size, sent_size)
    return vec_dictionary




def my_dictionary():
    '''
    bad and good (text) are stored in files
    '''
    sent_size = 16
    ru_dict_source = 'softlink_ru'
    en_dict_source = 'softlink_en'
    dict_ready = 'dictionary-ru'
    ready_bad = 'ready_bad'
    ready_good = 'ready_good'
    
    dictionary, vec_size = get_dict(ru_dict_source)
    vec_dictionary_bad = prepare_corpora(ready_bad,  \
                                         dictionary, \
                                         vec_size,   \
                                         sent_size)
    vec_dictionary_good = prepare_corpora(ready_good, \
                                          dictionary, \
                                          vec_size,   \
                                          sent_size)
    
    return vec_dictionary_bad, \
        vec_dictionary_good, vec_size


#     corpora_bad = 'corpora_bad'
#     corpora_good = 'corpora_good'
#     ready_bad = 'ready_bad'
#     ready_good = 'ready_good'
#     ready_data_good = prepare_corpora(corpora_good)
#     ready_data_bad = prepare_corpora(corpora_bad)
#     store_data(ready_data_bad, ready_bad)
#     store_data(ready_data_good, ready_good)    
#     
#     bad.close()
#     good.close()
#     return res_bad, res_good
