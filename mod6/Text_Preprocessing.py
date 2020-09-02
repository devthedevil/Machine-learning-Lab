import numpy as np
import pandas as pd
import string
import nltk
from sklearn.datasets import load_files
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.chunk import RegexpParser
from nltk.util import ngrams

categories = ['alt.atheism']
twenty_news_data = load_files(container_path = '20_newsgroups', categories = categories, random_state = 42)
strings  = twenty_news_data.data[0]
categories = ['rec.motorcycles']
twenty_news_data = load_files(container_path = '20_newsgroups', categories = categories, random_state = 56)
strings  += twenty_news_data.data[0]
categories = ['sci.electronics']
twenty_news_data = load_files(container_path = '20_newsgroups', categories = categories, random_state = 56)
strings  += twenty_news_data.data[0]

words_string = strings.decode("utf-8")
word_string = words_string.lower()
print(word_string)

words = word_tokenize(word_string)
print(words)

#Punctuation Removal
for w in string.punctuation:
    for s in words:
        if w==s:
            words.remove(s)

print(words)

#Stemming and Lemmatiaztion
stemmed_data = []
for w in words:
    stemmed_data.append(PorterStemmer().stem(w))
print(stemmed_data)

lemmed_data = []
for w in stemmed_data:
    lemmed_data.append(WordNetLemmatizer().lemmatize(w))
print(lemmed_data)

#stopword 
stopping_data=set(stopwords.words('english'))
textdata = [w for w in lemmed_data if not w in stopping_data]
print(textdata)


#Sentence tokenization
sentences = nltk.tokenize.sent_tokenize(word_string)
print(sentences)

#Removing Stopwords and Punctuation

sentence_token = [w for w in sentences if not w in stopping_data]
sentence_token = [''.join(c for c in s if c not in string.punctuation) for s in sentence_token]
sentence_token = [s for s in sentence_token if s]
print(sentence_token)


#POS Tagging, Chunking and N-grams
def extract_ngrams(data, num):
    n_grams = ngrams(word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]


for t in sentence_token:
    #POS_Tagging
    print(t)
    wordsList = word_tokenize(t) 
    pos_tagged=pos_tag(wordsList)
    print("After POS-Tagging\n")
    print(pos_tagged)
    
    #Chunking
    chunker = RegexpParser(r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}""")
    output = chunker.parse(tagged)
    print("After chunking",'\n')
    print(output)
    
    #3-grams
    print("3 grams : ");
    print(extract_ngrams(t,3))
    