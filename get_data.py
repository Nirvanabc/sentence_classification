from bs4 import BeautifulSoup
import urllib.request
from functools import reduce
from re import findall
import pickle
from nltk.tokenize import sent_tokenize

newline = "!caret_return!"

def parse_reviews(url, pages):
    ''' 
    returns all reviews on given grade (bad or good)
    as a list of text corpuses
    '''
    corpora = []
    for i in range(1, pages):
        curr_url = url.replace('*', str(i))
        html = urllib.request.urlopen(curr_url).read()
        soup = BeautifulSoup(html, "html5lib")
        text = soup.findAll('span', class_= "_reachbanner_")

        a = str(text)
        a = a.replace("\n", newline)
        a = a.replace("<br/>", "")
        a = a.replace("\xa0", " ")
        a = a.replace("<b>", " ")
        a = a.replace("</b>", " ")
        a = a.replace("</span>", " ")
        a = a.replace(">", " ")
        a = a.replace("<i", " ")
        a = a.replace("</i", " ")
        a = a.replace('<span class="_reachbanner_" itemprop="', " ")
        a=a.split('reviewBody')
        a = a[1:]
        corpora += a        
    return corpora


def make_corpora():
    '''
    stores data in pickle files.
    '''
    file_corpora_bad = open('corpora_bad', 'wb')
    file_corpora_good = open('corpora_good', 'wb')
    url_bad = 'https://www.kinopoisk.ru/reviews/type/comment/status/bad/period/month/page/*/#list'
    url_good = 'https://www.kinopoisk.ru/reviews/type/comment/status/good/period/month/page/*/#list'
    pages_bad = pages_good = 31
    corpora_bad = parse_reviews(url_bad, pages_bad)
    corpora_good = parse_reviews(url_good, pages_good)
    pickle.dump(corpora_bad, file_corpora_bad)
    pickle.dump(corpora_good, file_corpora_good)
    

make_corpora()
