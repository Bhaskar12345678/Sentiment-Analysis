# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22, 2020

@author: Bhaskar Mahna

Supervised ML using classifiers - Logistic, SGDClassifier, SVN and NB 


"""


import pandas as pd
import numpy as np
from normalisation import normalize_corpus
from utils import build_feature_matrix




dataset = pd.read_csv(r'movie_reviews.csv')

print( dataset.head())

train_data = dataset[:35000]
test_data = dataset[35000:]

train_reviews = np.array(train_data['review'])
train_sentiments = np.array(train_data['sentiment'])


test_reviews = np.array(test_data['review'])
test_sentiments = np.array(test_data['sentiment'])


sample_docs = [100, 5817, 7626, 7356, 1008, 7155, 3533, 13010]
sample_data = [(test_reviews[index],
                test_sentiments[index])
                  for index in sample_docs]

sample_data1 = [test_reviews[index] for index in sample_docs]

print(sample_data1[2])

sample_data    



from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from normalisation import normalize_corpus
from sklearn.naive_bayes import MultinomialNB





# SGDClassifier Algo
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)),])


# Logistic Regression Algorithm saving
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', LogisticRegression(penalty='l2', max_iter=100, C=1)),])


# Naive Bias Algo
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])


norm_train_reviews = normalize_corpus(train_reviews,
                                      lemmatize=True,
                                      only_text_chars=True)


text_clf.fit(norm_train_reviews, train_data['sentiment'])




# Save the above model using pickle 
pickle.dump(text_clf , open('MultinomialNB_Classifier_With_Vectorizer_pipeline.pickle', 'wb'))




# Load the complete model with vectorizer, feature matrix and model values to preduict the text outcome
text_clf = pickle.load(open('LogisticRegression_Classifier_With_Vectorizer_pipeline.pickle', 'rb'))



test1 = ["Hi this is bad movies"]
predicted_sentiment =  text_clf.predict(test1)


predicted_sentiment =  text_clf.predict(test_data['review'])


print(predicted_sentiment)
print(test_data['review'])






#norm_train_reviews = normalize_corpus(sample_data1,
#                                      lemmatize=True,
#                                      only_text_chars=True)

#vectorizer, sample_features = build_feature_matrix(documents=norm_train_reviews,
                                                  feature_type='tfidf',
                                                  ngram_range=(1, 1), 
                                                  min_df=0.0, max_df=1.0)                                      


# normalization
norm_train_reviews = normalize_corpus(train_reviews,
                                      lemmatize=True,
                                      only_text_chars=True)



# feature extraction using tfidf with unigram                                                                            
vectorizer, train_features = build_feature_matrix(documents=norm_train_reviews,
                                                  feature_type='tfidf',
                                                  ngram_range=(1, 1), 
                                                  min_df=0.0, max_df=1.0)                                      
                                      
                                      
print(train_features)
# Import SGDClassified and LogisticRegression models for training and testing to see the results 
from sklearn.linear_model import SGDClassifier, LogisticRegression

# Build the model 
svm = SGDClassifier(loss='hinge', n_iter=500)


# Train the model on training set 
svm.fit(train_features, train_sentiments)


import pickle



from sklearn.pipeline import make_pipeline


pipeline = make_pipeline(vectorizer, svm)


# Saving complete pipeline after training and vectorizing the text corpus 
pickle.dump(pipeline, open('SGD_Classifier_With_Vectorizer_pipeline.pickle', 'wb'))



# Loading pipeline to predict the text directly using vectorizer and model saved earlier
pipeline1 = pickle.load(open('SGD_Classifier_With_Vectorizer_pipeline.pickle', 'rb'))

predicted_sentiment =  pipeline1.predict(norm_test_reviews)


# normalize reviews                        
norm_test_reviews = normalize_corpus(test_reviews,
                                     lemmatize=True,
                                     only_text_chars=False)  


# extract features                                     
test_features = vectorizer.transform(norm_test_reviews)         


print(test_features)
# Predict sentiments of Sample Data
for doc_index in sample_docs:
    print ('Review:-')
    print (test_reviews[doc_index])
    print ('Actual Labeled Sentiment:', test_sentiments[doc_index])
    doc_features = test_features[doc_index]
    predicted_sentiment = svm.predict(doc_features)[0]
    print ('Predicted Sentiment:', predicted_sentiment)
    print
   

# Predict the model on Test set
predicted_sentiments = svm.predict(test_features)       

print(predicted_sentiments)

# See the results metrices
from utils import display_evaluation_metrics, display_confusion_matrix, display_classification_report

display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=predicted_sentiments,
                           positive_class='positive')  
                           
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=predicted_sentiments,
                         classes=['positive', 'negative'])
                         
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=predicted_sentiments,
                              classes=['positive', 'negative']) 




'''
Accuracy: 0.89
Precision: 0.88
Recall: 0.9
F1 Score: 0.89



                 Predicted:         
                   positive negative
Actual: positive       6778      732
        negative        916     6574



             precision    recall  f1-score   support

   positive       0.88      0.90      0.89      7510
   negative       0.90      0.88      0.89      7490

avg / total       0.89      0.89      0.89     15000                        

'''

'''
------------- Logistic Regression

'''

import model_evaluation_utils as meu


lr = LogisticRegression(penalty='l2', max_iter=100, C=1)


lr_bow_predictions = meu.train_predict_model(classifier=lr, train_features= train_features, 
                                             train_labels=train_sentiments, test_features=test_features, 
                                             test_labels=test_sentiments)

print(test_features)
print(test_sentiments)
print(lr_bow_predictions)
--------------------------------------------------------------------------------------------------
Actual Senitments    -    ['negative' 'positive' 'negative' ... 'negative' 'negative' 'negative']
Predicted Sentiments -    ['negative' 'positive' 'negative' ... 'positive' 'negative' 'negative']
----------------------------------------------------------------------------------------------------
  
meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=lr_bow_predictions, 
                                      classes=['positive', 'negative'])


'''
Model Performance metrics:
------------------------------
Accuracy: 0.8897
Precision: 0.8899
Recall: 0.8897
F1 Score: 0.8897

Model Classification report:
------------------------------
             precision    recall  f1-score   support

   positive       0.88      0.90      0.89      7510
   negative       0.90      0.88      0.89      7490

avg / total       0.89      0.89      0.89     15000


Prediction Confusion Matrix:
------------------------------
                 Predicted:         
                   positive negative
Actual: positive       6756      754
        negative        900     6590

'''

'''
------------ SVM - Support Vector Machine

'''
import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import svm

    data_dir = "C:\Bhaskar Sem 8\Capstone project\Capstome Sem 8 Documents submitted to College\Sentiment_Analysis\data"
    classes = ['pos', 'neg']


  # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:
                content = f.read()
                if fname.startswith('cv9'):
                    test_data.append(content)
                    test_labels.append(curr_class)
                else:
                    train_data.append(content)
                    train_labels.append(curr_class)
                    

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)


    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

                    
                    
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)



    t1 = time.time()
    
    prediction_rbf = classifier_rbf.predict(test_vectors)
    
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1-t0
    time_liblinear_predict = t2-t1

    # Print results in a nice table
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))




'''
-----------------  Naive Bais
'''


import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


'''
  # Read the data
train_data = dataset[:35000]
test_data = dataset[35000:]

train_reviews = np.array(train_data['review'])
train_lables = np.array(train_data['sentiment'])
test_reviews = np.array(test_data['review'])
test_lables = np.array(test_data['sentiment'])
'''


    data_dir = "C:\Bhaskar Sem 8\Capstone project\Capstome Sem 8 Documents submitted to College\Sentiment_Analysis\data"
    classes = ['pos', 'neg']



    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:
                content = f.read()
                if fname.startswith('cv9'):
                    test_data.append(content)
                    test_labels.append(curr_class)
                else:
                    train_data.append(content)
                    train_labels.append(curr_class)
                    
             
                    



    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)

'''
    # Corrected the following issue as  
    # Sounds to me, like you just need to use vectorizer.transform for the test dataset, 
    # since the training dataset fixes the vocabulary (you cannot know the full vocabulary including 
    # the training set afterall). Just to be clear, thats vectorizer.transform instead of 
    # vectorizer.fit_transform
    # test_vectors = vectorizer.fit_transform(test_data)
'''
    test_vectors = vectorizer.transform(test_data)


print(train_vectors)
print(test_vectors)
#    train_vectors = vectorizer.fit_transform(train_data).astype(float)
#    test_vectors = vectorizer.fit_transform(test_data).astype(float)

    clf = MultinomialNB()
    t0 = time.time()
    clf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction = clf.predict(test_vectors)


    t2 = time.time()
    time_train = t1-t0
    time_predict = t2-t1


    # Print results in a nice table
    print("Results for NaiveBayes (MultinomialNB) ")
    print("Training time: %fs; Prediction time: %fs" % (time_train, time_predict))
    print(classification_report(test_labels, prediction))
    print ("Reviews Prediction")
    print (prediction[10] + "----"+test_data[10])

    print ("\nReviews Prediction")
    print (prediction[100] + "----" + test_data[100])


import model_evaluation_utils as meu


display_evaluation_metrics(true_labels=test_labels,
                           predicted_labels=prediction,
                           positive_class='positive')  
                           
display_confusion_matrix(true_labels=test_labels,
                         predicted_labels=prediction,
                         classes=['positive', 'negative'])
                         
display_classification_report(true_labels=test_labels,
                              predicted_labels=prediction,
                              classes=['positive', 'negative']) 



    
    meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=prediction, 
                                      classes=['positive', 'negative'])

    
    '''
                 precision    recall  f1-score   support

        neg       0.81      0.92      0.86       100
        pos       0.91      0.78      0.84       100

avg / total       0.86      0.85      0.85       200
    '''
    
    