# -*- coding: utf-8 -*-
"""
Created on April  10 15:51:11 2020

@author: Bhaskar Mahna

 - Using Word Embedding of Trained Glove for Feature extraction and use our text using Glove embedding to provide into 
Keras layer 

LSTM model

"""


import os

imdb_dir = 'C:/Bhaskar Sem 8/Capstone project/Movie Sentiment Analysis using Keras - Deep Learning/aclImdb/'
train_dir = os.path.join(imdb_dir, 'train')

print(train_dir)

labels = []
texts = []

for label_type in ['neg', 'pos']:
    #dir_name = os.path.join(train_dir, label_type)
    dir_name = train_dir + '/'+label_type
    print(dir_name)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), 'r', encoding='utf8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)    
                
                
print(texts)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

import model_evaluation_utils as meu

maxlen = 100                                           #1
training_samples = 200                                 #2
validation_samples = 10000                             #3
max_words = 10000    

                
tokenizer = Tokenizer(max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)


labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])                     #5
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

"""
1 Cuts off reviews after 100 words
2 Trains on 200 samples
3 Validates on 10,000 samples
4 Considers only the top 10,000 words in the dataset
5 Splits the data into a training set and a validation set, but first shuffles the data, 
because you’re starting with data in which samples are ordered (all negative first, then all positive)
"""
import io
glove_dir = 'C://Bhaskar Sem 8/Capstone project/Movie Sentiment Analysis using Keras - Deep Learning/'

embeddings_index = {}
#f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
f = io.open('C://Bhaskar Sem 8/Capstone project/Movie Sentiment Analysis using Keras - Deep Learning/glove.6B.100d.txt', 'r', encoding='utf8')
#with io.open(filename,'r',encoding='utf8') as f:
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#print(embeddings_index)

# Preparing the GloVe word-embeddings matrix

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector        


# Define model
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Loading pretrained word embeddings into the Embedding layer
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

# C:\Bhaskar Sem 8\Capstone project\Movie Sentiment Analysis using Keras - Deep Learning
