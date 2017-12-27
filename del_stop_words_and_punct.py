import string
from sklearn.feature_extraction import stop_words


def del_punct_and_stopwords(sent):
    tmp_result = ''
    result = []
    for letter in sent:
        if not letter in string.punctuation:
            tmp_result += letter
    tmp_result = tmp_result.split()
    return tmp_result
#     for word in tmp_result:
#         if not word in stop_words.ENGLISH_STOP_WORDS:
#             result.append(word)
#     return result


def clear_corpora(corpora_from, corpora_to):
    result = []
    for sent in corpora_from:
        result.append(del_punct_and_stopwords(sent))
    for sent in result:
        for word in sent[:-1]:
            corpora_to.write(word + ' ')
        corpora_to.write(sent[-1])
        corpora_to.write("\n")


def make_corpora():
    train = open('train_MR')
    test = open('test_MR')
    res_train = open('new_train_MR', 'w')
    res_test = open('new_test_MR', 'w')
    clear_corpora(train, res_train)
    clear_corpora(test, res_test)
