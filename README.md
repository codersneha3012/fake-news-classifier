# Fake news classifier

Download dataset from [Kaggle-Fake News Classifier](https://www.kaggle.com/c/fake-news)

## Machine learning pipeline for NLP Problems is created using Sklearn 

#### Solution 1: CountVectorizer & Naive Bayes. Hyperparameter tuned using GridSearchCV.

Accuracy - 89%
![CM Plot](static/plt1.png)


#### Solution 2: CountVectorizer & PassiveAggressiveClassifier. Hyperparameter tuned using GridSearchCV.

Accuracy - 93%
![CM Plot](static/plt2.png)


#### Solution 3: WordEmbedding

One lstm layer with 100 neurons
Epoch 10/10
222/222 [==============================] - 4s 19ms/step - loss: 2.6340e-04 - accuracy: 0.9999 - val_loss: 0.5344 - val_accuracy: 0.9206
Accuracy: 0.9206323069323235

Two lstm layers with 50 neurons W/o Dropout
Epoch 10/10
222/222 [==============================] - 5s 20ms/step - loss: 0.0039 - accuracy: 0.9989 - val_loss: 0.4144 - val_accuracy: 0.9216
Accuracy: 0.9216202865140787
Over-fitting

Two lstm layers with 50 neurons With Dropout
Accuracy: 0.921290959986827
Performs better, but in this case not that significant difference