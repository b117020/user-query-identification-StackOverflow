# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 01:47:37 2019

@author: Devdarshan
"""
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from ast import literal_eval
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from collections import Counter


def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data




train = read_data('train.tsv')
validation = read_data('validation.tsv')
test = pd.read_csv('test.tsv', sep='\t')


train.head()

X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values



nltk.download('punkt')



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]') 
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]') # take all words that contain characters other than 0-9,a-z,#,+
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    #text = # lowercase text
    text =text.lower()
    #text = # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    #text = # delete symbols which are in BAD_SYMBOLS_RE from text
    text =  re.sub(BAD_SYMBOLS_RE, '', text)
    #text = # delete stopwords from text
    token_word=word_tokenize(text)
    filtered_sentence = [w for w in token_word if not w in STOPWORDS] # filtered_sentence contain all words that are not in stopwords dictionary
    lenght_of_string=len(filtered_sentence)
    text_new=""
    for w in filtered_sentence:
        if w!=filtered_sentence[lenght_of_string-1]:
             text_new=text_new+w+" " # when w is not the last word so separate by whitespace
        else:
            text_new=text_new+w
            
    text = text_new
    return text
'''
def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

print("done")

prepared_questions = []
for line in open('text_prepare_tests.tsv', encoding='utf-8'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)
text_prepare_results = '\n'.join(prepared_questions)


print("done")
'''

X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]
print("done")


print(len(X_train))

#import collections 


words=[]
tag_w=[]
for i in range(0,100000):
    print(i)
    words = words+(re.findall(r'\w+', X_train[i])) # words cantain all the words in the dataset
    tag_w=tag_w+y_train[i] # tage_w contain all tags that aree present in train dataset
    
words_counts = Counter(words) # counter create the dictinary of unique words with their frequncy
tag_counts=Counter(tag_w)
#print(words_counts)
#print(tag_counts)


from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    tfidf_vectorizer =  TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')#  '(\S+)'  means any no white space
    X_train=tfidf_vectorizer.fit_transform(X_train)
    X_val=tfidf_vectorizer.transform(X_val)
    X_test=tfidf_vectorizer.transform(X_test)
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}

print('X_test_tfidf ', X_test_tfidf.shape) 
print('X_val_tfidf ',X_val_tfidf.shape)

print(tfidf_vocab)


from sklearn.preprocessing import MultiLabelBinarizer


mlb = MultiLabelBinarizer(classes=sorted(tag_counts.keys()))
y_train = mlb.fit_transform(y_train) # it chnage the y_train in feature form like alll clases with 0,1 value

y_val = mlb.fit_transform(y_val)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    model=OneVsRestClassifier(LogisticRegression()).fit(X_train,y_train)
    
    return model

    ######################################
    ######### YOUR CODE HERE #############
    ######################################   
    
print('X_test_tfidf ', X_test_tfidf.shape) 
print('X_val_tfidf ',X_val_tfidf.shape)


classifier_tfidf = train_classifier(X_train_tfidf, y_train)
'''
def clf_predict():
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf) # just opposite of tranform means it will give the name of classes rather than 0,1 in classes
    y_val_inversed = mlb.inverse_transform(y_val)
    return y_val_pred_inversed
'''
def manual_predict(X_test):
    X_test = [text_prepare(x) for x in X_test]
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_test_tfidf)
    y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
    for i in range(1):
        stored = y_val_pred_inversed[i]
    return stored
'''
query = input("enter query: ")
X_test.insert(0,query)
X_test = [text_prepare(x) for x in X_test]
X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_test_tfidf)
y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
'''
# importing the requests library 

import requests 
def ques_pred():
    tag = manual_predict(X_test)
    for i in range(1):
        URL = "http://api.stackexchange.com/2.2/questions?page=1&pagesize=5&todate=1566777600&order=asc&sort=votes&tagged={}&site=stackoverflow".format(';'.join(tag[i]))

    

# api-endpoint 
# ds = "java;kotlin" 


# sending get request and saving the response as response object 
    r = requests.get(url = URL)

# extracting data in json format 
    data = r.json()
    title = []
    link = []
# print(data)
    
    for item in data['items']:
        title[item] = item['title']
        link[item] = item['link']
    return title,link
  

    