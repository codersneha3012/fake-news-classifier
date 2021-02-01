#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:55:54 2021

@author: sneha
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from news_preprocessing import preclean, plot_confusion_matrix, tokenize

df = pd.read_csv('fake-news/train.csv')

df.head()

df_copy = df.copy()

preclean(df_copy)
X = df_copy['title']
y = df_copy['label']

voc_size=10000
sent_len = 20
embedding_vector_features = 40

#one-hot representation
onehot_repr = [one_hot(words, voc_size) for words in X]
corpus = pad_sequences(onehot_repr, maxlen=sent_len)

y = np.array(y)
# train-test split
X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.3, random_state=42)

model1 = Sequential()
model1.add(Embedding(voc_size, embedding_vector_features, input_length=sent_len))
model1.add(LSTM(10))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #, run_eagerly=True)

earlystopping_callback = EarlyStopping()
model1.fit(X_train, y_train,validation_data=(X_test,y_test)
           ,epochs=10,batch_size=64,
           callbacks=[earlystopping_callback])

y_pred = model1.predict_classes(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])

##############################
# model2
# model2 = Sequential()
# model2.add(Embedding(voc_size, embedding_vector_features, input_length=sent_len))
# model2.add(Dropout(0.3))
# model2.add(LSTM(25, return_sequences=True, name='lstm1'))
# model2.add(Dropout(0.3))
# model2.add(LSTM(25, name='lstm2'))
# model2.add(Dropout(0.3))
# model2.add(Dense(1, activation='sigmoid'))
# model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model2.fit(X_train, y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

# y_pred = model2.predict_classes(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# cm = confusion_matrix(y_test, y_pred)
# plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])








