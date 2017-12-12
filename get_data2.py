'''
stores pickle data in txt format
'''

import pickle
from random import shuffle

def label_corpora(corpora, label):
    for sent in corpora:
        sent.append(label)


def shuffle_data(file_bad, file_good, file_to):
    label_bad = 0
    label_good = 1
    res_file = open(file_to, 'w')
    with open(file_bad, 'rb') as f:
        bad = pickle.load(f)
    with open(file_good, 'rb') as f:
        good = pickle.load(f)
    label_corpora(bad, label_bad)
    label_corpora(good, label_good)
    data = bad + good
    shuffle(data)
    
    new_data = open(file_to, 'w')
    for sent in data:
        if len(sent) > 1:
            for word in sent:
                new_data.write(str(word))
                new_data.write(" ")
            new_data.write("\n")
    
    return data


def make_corpora():
    bad = 'ready_bad'
    good = 'ready_good'
    corpora = 'corpora_text'
    
    shuffle_data(bad, good, corpora)
    
