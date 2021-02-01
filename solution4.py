#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:55:54 2021

@author: sneha
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from news_preprocessing import preclean, plot_confusion_matrix, tokenize

df = pd.read_csv('fake-news/train.csv')

df.head()

df_copy = df.copy()

preclean(df_copy)
X = df_copy['title']
y = df_copy['label']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=30)

# create pipeline
model_pipeline = Pipeline(steps=[
        ('vect', CountVectorizer(tokenizer=tokenize, max_features=5000, ngram_range=(1,3))),
        # ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression())
    ], verbose=True)

parameters = {
    'clf__C': (0.1, 0.01, 0.001, 0.00001),
    'clf__penalty': ('l1', 'l2')
}
grid_search = GridSearchCV(model_pipeline, parameters, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])






