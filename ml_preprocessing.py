# -*- coding: utf-8 -*-

import json
import re
import math
import nltk
from io import open
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
lm = nltk.stem.WordNetLemmatizer()
tokenizr = TweetTokenizer()
from sklearn import metrics
from sklearn.naive_bayes import  MultinomialNB,GaussianNB,ComplementNB
import string, random, time, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import SGDClassifier
# Instantiates the device to be used as GPU/CPU based on availability
device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#################################### CHANGE PATH HERE ################################################
path = ""
embeddings_index={}
with open(path +'glove.twitter.27B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word = values[0]
        vectors=np.asarray(values[1:],'float32')
        embeddings_index[word]=vectors
f.close()

def y_encode(sentiment):
    sentmnt_map = {'negative': 0,'neutral': 1,'positive' : 2}
    return sentmnt_map[sentiment]

# can add other meta pre processing functions here.

def toknz_lematz(twts):
  stop = stopwords.words('english')
  tk_men_words = [lm.lemmatize(words) for words in tokenizr.tokenize((twts)) if words not in stop]
  return " ".join(tk_men_words)

def txt_preprocessor(twts):
  import re
  # remocve hashtags and users marked
  story = re.sub(r"[#@0-9](\w+)", "",twts)

  story = re.sub(r"[^A-Za-z0-9 ]", "", story)
  # #remove single word character
  story = re.sub(r"\b\w\b", "", story)
  # remove all words with numbers
  story = re.sub(r'\w*\d\w*', '', story)
  # # remove single digits
  story = re.sub(r"\b\d+\b", "", story)
  # # remove links
  story = re.sub(r"\b(https?)[\w/.:?$@!]+\b", "", story)
  #again
  story = re.sub(r"[#@0-9](\w+)", "",story)
  # tokenize lemmatize
  story_1 = toknz_lematz(story.lower())
  
  return story

def get_glove_embed(sent,embedding_dict):
    tokens = sent.split(" ")
    ls  = []
    for i in tokens:
        try:  ls.append(embedding_dict[i])
        except: ls.append(np.zeros(100))
    return np.mean(np.array(ls),axis=0)

def glove_vect(df,embedding_dict):
    glove_embed = np.vstack(df.cln_twts.apply(lambda x : get_glove_embed(x,embedding_dict)).values)
    return glove_embed

#['CountVectorizer', 'TFIDF','Glove']
def preprocessor(df,classifier,features,dataset_type):
  
  if dataset_type == 'train':
    feat_vect = 0
    train_df = df
    train_df['cln_twts'] = train_df['text'].apply(lambda x: txt_preprocessor(x))
    train_y = train_df.sentiment.apply(y_encode) 

    # for machine learning approaches to get train_x 
    if classifier in ['SVM', 'Naive Bayes','Compliment NB','Gaussian NB']:

      if features == 'CountVectorizer':
        feat_vect = CountVectorizer(ngram_range=(1,2),max_features=1000)
        train_x = feat_vect.fit_transform(train_df.cln_twts)
        if classifier == "Gaussian NB":
          train_x = train_x.toarray()
        
      elif features == 'TFIDF':
        feat_vect = TfidfVectorizer(ngram_range=(1,2),max_features=1000)
        train_x = feat_vect.fit_transform(train_df.cln_twts)
        if classifier == "Gaussian NB":
          train_x = train_x.toarray()

      else : # Glove

        embedding_dict=embeddings_index
        # with open('/content/drive/MyDrive/Colab Notebooks/cs918/assg-2/glove.twitter.27B.100d.txt','r') as f:
        #   for line in f:
        #     values=line.split()
        #     word = values[0]
        #     vectors=np.asarray(values[1:],'float32')
        #     embedding_dict[word]=vectors
        # f.close()

        train_x = glove_vect(train_df,embedding_dict)
      return train_x,train_y, feat_vect

def model_trainer(x,y,classifier):
  if classifier == "Naive Bayes":
      clf1 = MultinomialNB()
  elif classifier == "SVM":
      clf1 = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)
  elif classifier == "Compliment NB":
      clf1 = ComplementNB()
  elif classifier =="Gaussian NB":
      clf1 = GaussianNB()
  clf1.fit(x,y)
  return clf1

def predictor(df,model,feat_vect,classifier,features,dataset_type):

  if classifier in ['SVM', 'Naive Bayes','Compliment NB','Gaussian NB']:
    test_df = df
    test_df['cln_twts'] = test_df['text'].apply(lambda x: txt_preprocessor(x))
    test_y = test_df.sentiment.apply(y_encode)
    
    if features != 'Glove':  
      
      test_x = feat_vect.transform(test_df.cln_twts)
      if classifier == "Gaussian NB":
        pred = list(model.predict(test_x.toarray()))
      else: 
        pred = list(model.predict(test_x))
      pred_dict = dict(zip(list(df.id.values),pred))

    else:
      embedding_dict = embeddings_index
      # embedding_dict={}
      # with open('/content/drive/MyDrive/Colab Notebooks/cs918/assg-2/glove.twitter.27B.100d.txt','r') as f:
      #   for line in f:
      #     values=line.split()
      #     word = values[0]
      #     vectors=np.asarray(values[1:],'float32')
      #     embedding_dict[word]=vectors
      # f.close()

      test_x = glove_vect(test_df,embedding_dict)
      pred = list(model.predict(test_x))
#       pred_dict = dict(zip(list(df.id.values),pred))      
#   f1 = metrics.f1_score(pred,test_y, average=None)
#   f1_macro = (f1[0]+f1[2])/2
#   print(f"f1 score for Test{i+1}: {f1_macro}")
  return pred,test_y
