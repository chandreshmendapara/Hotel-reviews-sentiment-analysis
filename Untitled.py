#!/usr/bin/env python
# coding: utf-8

# In[1]:


5+3


# In[1]:


import simplejson as json
from matplotlib import pyplot as plt 
from langdetect import detect
import re
import pandas as pd 
import numpy as np 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)



test = pd.read_csv('7282_1.csv')
test.head(2)


# In[2]:



print(test.shape)
test.keys()
hotels=test[['name','reviews.date','reviews.rating','reviews.text']].drop_duplicates()
print(hotels.shape)

#hotels.reset_index()
hotels.reset_index(drop=True)


# In[53]:


#hotels['tidy_tweet'] = hotels['reviews.text'].str.replace("[^a-zA-Z#]", " ")
#hotels.head(100)


# In[25]:


hotels=hotels.dropna()
hotels['tidy_tweet'] = hotels['reviews.text'].str.replace("[^a-zA-Z#]", " ")
hotels['name'] = hotels['name'].str.replace("Hotel", "")

#hotels.head(100)
hotels['tidy_tweet'] = hotels['reviews.text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
#df['subject acc.ver'] = df['subject acc.ver']
hotels.head()


# In[4]:


#tokenized_tweet = hotels['tidy_tweet'].apply(lambda x: x.split())
#tokenized_tweet.head()
l1=[]
hotels['tokenized_tweet'] = hotels.apply(lambda x: x['tidy_tweet'].split(),axis=1)
#hotels['tokenized_tweet'] = hotels['tidy_tweet'].apply(lambda x: x.split())
#hotels['tokenized_tweet'].head()
hotels.head()


# In[5]:


"""from nltk.stem.porter import *
stemmer = PorterStemmer()
#stemming
hotels['stemming'] = hotels['tokenized_tweet'].apply(lambda x: [stemmer.stem(i) for i in x])
hotels.head()"""
tokenized_tweet = hotels['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()


# In[40]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

hotels['tokenized_tweet'] = hotels['tokenized_tweet'].apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
hotels['tokenized_tweet'].head()
ls=[]
ls=hotels['tokenized_tweet']

hotels['tidy_tweet'] = hotels['tokenized_tweet'].apply(lambda x: ' '.join(map(str,x)))
hotels['tidy_tweet'].head()


# In[ ]:




