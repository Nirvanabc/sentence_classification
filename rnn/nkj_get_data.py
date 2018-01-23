import urllib.request
from bs4 import BeautifulSoup
import ssl
import re

def download_articles(url, first_page, last_page):
    articles = []
    article_count = 0
    context = ssl._create_unverified_context() 
    for i in range(first_page, last_page):
        curr_url = url.replace('*', str(i))
        html = urllib.request.urlopen(curr_url, \
                                      context=context).read() 
        soup = BeautifulSoup(html, "html5lib")
        text = soup.findAll('div', class_= "text")
        print(i, article_count)
        if text == []: continue
        article_count += 1
        a = str(text[0])
        b = re.sub('[a-zA-Z]|<|>|/|=|"|\+|[0-9]|\&|\%|\_', '', a)
        b = re.sub("\n", " ", b)
        b = re.sub("\W\W\W\W+", " ", b)
        b = re.sub(" +", " ", b)
        articles += [b]
    return articles, article_count

def make_corpora():
    file_corpora = open("input.txt", 'w')
    url = 'https://www.nkj.ru/archive/articles/*'
    first_page = 27712
    last_page = 33045
    articles, article_count = download_articles(url, \
                                                first_page, \
                                            last_page)
    for article in articles:
        file_corpora.write(article)
        file_corpora.write("\n")
    file_corpora.close()
