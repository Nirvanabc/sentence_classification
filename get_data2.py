import pickle
from constants import *
from random import shuffle

def label_corpora(corpora, label):
    for sent in corpora:
        sent.append(label)


def read_and_label_corpora(corpora, label):
    result = []
    while True:
        sent = corpora.readline()
        if sent == '':
            break
        sent = [sent] + [label]
        result.append(sent)
    return result


def shuffle_data(file_bad, file_good, file_to):
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


def shuffle_text_data(file_bad, file_good, file_to):
    res_file = open(file_to, 'w')
    bad = open(file_bad)
    good = open(file_good)
    result_bad = read_and_label_corpora(bad, label_bad)
    result_good = read_and_label_corpora(good, label_good)
    result = result_bad + result_good
    shuffle(result)
    for i in result:
        res_file.write(i[0][:-1])
        res_file.write(" ")
        res_file.write(str(i[1]))
        res_file.write("\n")
    res_file.close()
    bad.close()
    good.close()
    
        

def make_corpora():
    bad = 'neg_MR'
    good = 'pos_MR'
    corpora = 'corpora_MR'
    
    shuffle_text_data(bad, good, corpora)
