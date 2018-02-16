import numpy as np
from numpy.random import normal


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
    my_dict = open(dict_file, 'rb')
    line = my_dict.readline()
    line = line.split()
    row = int(line[0])
    col = int(line[1])
    result_dict = {}
    for _ in range(row):
        word, vec = read_word_and_its_vec(my_dict, col)
        result_dict[word] = vec
    my_dict.close()
    return result_dict, col


def normalize(vec):
    total = sum(vec)
    return list(map(lambda x: x/total, vec))


def word2vec(word, vec_size):
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
            # curr.append(word2vec(word, vec_size))
            # to test without softlink_ru uncomment the next
            # line and comment the previous
            curr.append(normalize(normal(size = vec_size)))
        result.append(curr)
    return result


def padd_sent(sent, vec_size, sent_size):
    res_sent = np.zeros([sent_size, vec_size])
    if len(sent) <= sent_size:
        res_sent[sent_size - len(sent):] = sent
    else: res_sent = sent[:sent_size]
    return res_sent


def padd_corpora(corpora, vec_size, sent_size):
    res_corpora = []
    for sent in corpora:
        res_corpora.append(padd_sent(sent, vec_size, sent_size))
    return res_corpora


def prepare_corpora(corpora, vec_size, sent_size):
    '''
    takes a batch and prepare it
    '''
    # corpora = del_empty(corpora)
    vec_dictionary = corpora2vec(corpora, vec_size)
    vec_dictionary = padd_corpora(vec_dictionary,
                                  vec_size,
                                  sent_size)
    return vec_dictionary
