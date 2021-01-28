#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:55:54 2021

@author: sneha
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from news_preprocessing import preclean, preprocess_pipeline, vectorizer_bow, plot_confusion_matrix, tokenize

df = pd.read_csv('fake-news/train.csv')

df.head()

df_copy = df.copy()

preclean(df_copy)
X = df_copy['title']
y = df_copy['label']

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=30)

# create pipeline
model_pipeline = Pipeline(steps=[
        ('vect', CountVectorizer(tokenizer=tokenize, max_features=5000, ngram_range=(1,3))),
        # ('tfidf', TfidfTransformer()),
        ('multinomial_nb', MultinomialNB(alpha=0.1))
    ], verbose=True)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])





