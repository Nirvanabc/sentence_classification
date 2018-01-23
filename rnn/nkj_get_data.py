import urllib.request
from bs4 import BeautifulSoup
import ssl
import re


def clean_text(text):
    ''' text must be str, returns str '''
    b = re.sub('[a-zA-Z]|<|>|/|=|"|\+|[0-9]|\&|\%|\_', '', text)
    b = re.sub("\n", " ", b)
    b = re.sub("\W\W\W\W+", " ", b)
    b = re.sub(" +", " ", b)
    return b


def write_data(data, file_to_write):
    ''' data is a list '''
    for article in data:
        file_to_write.write(article)
        file_to_write.write("\n")
        

def download_articles(url, first_page, last_page):
    file_corpora = open("input.txt", 'a')
    articles = []
    article_count = 0
    # without this line you can't get data, access denied.
    context = ssl._create_unverified_context() 
    for i in range(first_page, last_page):
        curr_url = url.replace('*', str(i))
        html = urllib.request.urlopen(curr_url, \
                                      context=context).read() 
        soup = BeautifulSoup(html, "html5lib")
        text = soup.findAll('div', class_= "text")
        if text == []: continue
        article_count += 1
        print(i, article_count)
        text = str(text[0])
        text = clean_text(text)
        articles += [text]
        if article_count % 100 == 0:
            write_data(articles, file_corpora)
            articles = []
    write_data(articles, file_corpora)
    file_corpora.close()

 
def make_corpora():
#     first_page = 27712
#     last_page = 33045
    url = 'https://www.nkj.ru/archive/articles/*'
    first_page = 3100
    last_page = 27711
    download_articles(url, first_page, last_page)
