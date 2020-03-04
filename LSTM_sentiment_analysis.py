# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:26:44 2019

@author: V15AShehata12
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re



tweet_files = glob.glob("./twitter-201?train.txt")

# Load train dataset into dataframe

li = []

for filename in tweet_files:
    df = pd.read_csv(filename, index_col=None, names=['Timestamp', 'Sentiment', 'Tweet'], sep='\t')
    li.append(df)

tweets_data = pd.concat(li, axis=0, ignore_index=True)

tweets_data = tweets_data[['Sentiment','Tweet']]

tweets_data['Tweet'] = tweets_data['Tweet'].apply(lambda x: x.lower())
tweets_data['Tweet'] = tweets_data['Tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

for idx,row in tweets_data.iterrows():
    row[0] = row[0].replace('rt',' ')
    
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(tweets_data['Tweet'].values)
X = tokenizer.texts_to_sequences(tweets_data['Tweet'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())









   