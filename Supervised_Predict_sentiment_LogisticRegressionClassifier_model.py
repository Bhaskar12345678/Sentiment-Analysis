# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:27:27 2020

@author: Bhaskar Mahna
"""

import sys
import pickle
from normalisation import normalize_corpus


def Supervised_Predict_sentiment_LogisticRegressionClassifier_model(review):
    
    rev = [] 
    rev.append(review)
    
    # Saved SGD Model
    
    #filename = ".\LogisticRegression_Classifier_With_Vectorizer_pipeline.pickle"
    
    filename = 'C:/Bhaskar Sem 8/Capstone project/Capstome Sem 8 Documents submitted to College/Sentiment_Analysis/LogisticRegression_Classifier_With_Vectorizer_pipeline.pickle';
    
    # Load the complete model with vectorizer, feature matrix and model values to preduict the text outcome
    text_clf = pickle.load(open(filename, 'rb'))


    norm_train_reviews = normalize_corpus(rev,
                                      lemmatize=True,
                                      only_text_chars=True)


    # Predict the text 
    predicted_sentiment =  text_clf.predict(norm_train_reviews)
    
    # print(predicted_sentiment)
    print('{"sentiment": "' + predicted_sentiment[0] + '"}')


if __name__ == '__main__':
# Map command line arguments to function arguments.
    Supervised_Predict_sentiment_LogisticRegressionClassifier_model(*sys.argv[1:])