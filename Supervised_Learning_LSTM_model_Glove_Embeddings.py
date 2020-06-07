# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 23:10:47 2020

@author: Bhaskar Mahna

LSTM model with Keras on TensorFlow
"""


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Flatten
from keras.layers import LSTM
from keras.datasets import imdb
import pandas as pd 
import numpy as np
from bs4 import BeautifulSoup
import re
import pickle
from keras.preprocessing.text import Tokenizer

#(x_train, y_train), (x_test, y_test) = imdb.load_data()


def save_predict_model(classifier, filename):
    filename = filename
    pickle.dump(classifier, open(filename, 'wb'))
    return filename



dataset = pd.read_csv(r'movie_reviews.csv')

print(dataset.head())

print(dataset['review'])

reviews = np.array(dataset['review'])
sentiments = np.array(dataset['sentiment'])


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

clean_docs = []

# can remove stopwords
for doc in reviews:
    doc = strip_html_tags(doc)
    doc = doc.lower().strip()
    doc = re.sub('[^a-zA-z0-9\s]','',doc)
    clean_docs.append(doc)
    
# Only 10000 most common words will be considered and ' ' - space will be consider for tokenizing
tokenizer = Tokenizer(num_words=10000, split=' ')

# Updates internal vocabulary based on a list of texts.
tokenizer.fit_on_texts(clean_docs)

# Get index of each word
tokenizer.word_index

# List of each review with index of each word in each review
X = tokenizer.texts_to_sequences(clean_docs)

# Padded the list based on max length review
X = sequence.pad_sequences(X)


'''

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



'''

# One hot encoding
# The pandas get_dummies() function is used to convert categorical variable into dummy/indicator variables.
# in our case Negative and Positive categorical values to
# Negative : [0,1]
# Positive: [1,0] 
sentiment_ohe = np.array(pd.get_dummies(sentiments))

sentiment_ohe


# Separate  first 35000 to Train the model and next 15000 to test the model
train_X = X[:35000]
train_y = sentiment_ohe[:35000]
test_X = X[35000:]
test_y = sentiment_ohe[35000:]

train_X.shape, train_y.shape, test_X.shape, test_y.shape

# Setting Hyper parameters
embed_dim = 128
lstm_out = 64
max_words =10000



model = Sequential()
'''
model.add(Embedding(input_dim, output_dim, input_length = X.shape[1]))

input_dim: This is the size of the vocabulary in the text data. For example, if your data is 
            integer encoded to values between 0-10, then the size of the vocabulary would be 11 words.
output_dim: This is the size of the vector space in which words will be embedded. 
            It defines the size of the output vectors from this layer for each word. 
            For example, it could be 32 or 100 or even larger. Test different values for your problem.
input_length: This is the length of input sequences, as you would define for any input layer of a Keras model. 
        For example, if all of your input documents are comprised of 1000 words, this would be 1000.
'''
model.add(Embedding(10000, embed_dim, input_length = X.shape[1]))

model.add(Dropout(0.2))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(2, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())



batch_size = 100
model.fit(train_X, train_y, epochs = 5, batch_size=batch_size, verbose=1)

'''
 - Save the model for future use - LSTM - Glove model

'''

    

# To save Keras model use model.save and model.load_model instead of pickle
model.save('./saved_models/model.h5')  # creates a HDF5 file 'model.h5'

#  model.save_weights() will only save the weights so if you need, you are able to apply them on a different architecture
#  mode.save() will save the architecture of the model + the the weights + the training configuration + the state of the optimizer
#

# model.save_weights('pre_trained_glove_model.h5')


# Similarly, loading the model is done like this:

from keras.models import load_model
model1 = load_model('./saved_models/model.h5')



r = model1.predict_classes(test_X)


a = np.array(pd.DataFrame(test_y).idxmax(axis=1))
a



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(a, r, target_names=['Negative', 'Positive']))

'''
             precision    recall  f1-score   support

   Negative       0.84      0.89      0.87      7490
   Positive       0.89      0.83      0.86      7510

avg / total       0.86      0.86      0.86     15000
'''

confusion_matrix(a, r)
 
''' 
array([[6687,  803],
       [1256, 6254]], dtype=int64)
    
'''
    
reviews[0]

'''
Print Review Text and Actual Vs. Predicted Sentiment results
'''
i=0
for r1 in r:
    print(reviews[i])
    if r1 == 0:
        print('Predicted - Negative')
        if a[i] == 0:
            print('Actual - Negative')
        else:
            print('Actual - Positive')
    else:
        print('Predicted - Positive') 
        if a[i] == 0:
            print('Actual - Negative')
        else:
            print('Actual - Positive')
    i +=1

reviews[10]
a[10]


The revelation here is Lana Turner dancing ability. Though she was known privately to be an excellent nightclub and ballroom dancer,
 Miss Turner rarely got the opportunity to demonstrate this ability on film.<br /><br />So, viewers take notice! Here, MGM were clearly
 still trying to determine in what direction they would develop the still young starlet, and were, 
 therefore, consigning her to everything from Andy Hardy to Doctor Kildaire.<br /><br />In Two Girls on Broadway,
 however, she is given an excellent opportunity to display her native rhythm and ability to shift tempo
 in the lavish production number, My Wonderful One, Lets Dance. This number, is conceived and filmed, as a sort
 of hybrid between a Busby Berkely style extravaganza and the sort of routines Hermes Pan was designing for 
 Astaire and Rogers at RKO.<br /><br />Thus, the number opens with George Murphy and Miss Turner depicted as 
 bar patrons (with full chorus) before a curtain of black lame wherein Mr. Murphy croons the number to Miss Turner. 
 Then the camera, (on a boom) pulls backward in a remarkable crane shot to reveal an enormous stage, and a 
 rotating set equipped with steps, columns, enclosures and sliding walls.<br /><br />From this point on, Murphy 
 and Turner execute a fast stepping variety of moods and attitudes, including lifts, spins, soft shoe, and ending 
 with an electrifying series of conjoined pirouettes that concludes with Murphy both lifting and rotating Turner with thrilling speed to a racing orchestra.<br /><br />All told a dizzying feat that proves Miss Turner was fully capable of more than holding her own as a dancer, though I daresay most of her admirers would balk at relinquishing her from her throne as the queen of melodrama.