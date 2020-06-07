# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 23:10:47 2020

@author: Bhaskar Mahna

LSTM model with Keras on TensorFlow
"""


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.datasets import imdb
import pandas as pd 
import numpy as np
from bs4 import BeautifulSoup
import re
from keras.preprocessing.text import Tokenizer

#(x_train, y_train), (x_test, y_test) = imdb.load_data()



dataset = pd.read_csv(r'movie_reviews.csv')

print(dataset.head())

reviews = np.array(dataset['review'])
sentiments = np.array(dataset['sentiment'])


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

clean_docs = []

for doc in reviews:
    doc = strip_html_tags(doc)
    doc = doc.lower().strip()
    doc = re.sub('[^a-zA-z0-9\s]','',doc)
    clean_docs.append(doc)
    
tokenizer = Tokenizer(num_words=10000, split=' ')
tokenizer.fit_on_texts(clean_docs)

tokenizer.word_index

X = tokenizer.texts_to_sequences(clean_docs)

X = sequence.pad_sequences(X)

sentiment_ohe = np.array(pd.get_dummies(sentiments))
sentiment_ohe

train_X = X[:35000]
train_y = sentiment_ohe[:35000]
test_X = X[35000:]
test_y = sentiment_ohe[35000:]

train_X.shape, train_y.shape, test_X.shape, test_y.shape

embed_dim = 128
lstm_out = 64

model = Sequential()
model.add(Embedding(10000, embed_dim, input_length = X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


batch_size = 100
model.fit(train_X, train_y, epochs = 5, batch_size=batch_size, verbose=1)

r = model.predict_classes(test_X)

a = np.array(pd.DataFrame(test_y).idxmax(axis=1))
a

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(a, r, target_names=['Negative', 'Positive']))

confusion_matrix(a, r)

reviews[0]




