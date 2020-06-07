# -*- coding: utf-8 -*-
"""
Created on Tue Feb 01 19:45:40 2020

@author: Bhaskar Mahna

Description: Unsupervised Machine Learning using various Lexicon

 Using Afin Lexicon dictionary library to score sentiment for movie data
 
 
Install Afinn library using
pip install Afinn in virtual env.

 
"""

import pandas as pd
import numpy as np



dataset = pd.read_csv(r'movie_reviews.csv')

print (dataset.head())

train_data = dataset[:35000]
test_data = dataset[35000:]

test_reviews = np.array(test_data['review'])
test_sentiments = np.array(test_data['sentiment'])


sample_docs = [100, 5817, 7626, 7356, 1008, 7155, 3533, 13010]
sample_data = [(test_reviews[index],
                test_sentiments[index])
                  for index in sample_docs]


sample_data        

'''

Install Afinn library using
pip install Afinn in virtual env.

'''
from afinn import Afinn
afn = Afinn(emoticons=True) 
print (afn.score('I really hated the plot of this movie'))

print (afn.score('I really hated the plot of this movie :('))


'''
Say you have a list of tuples and want to separate the elements of each tuple into independent sequences.
To do this, you can use zip() along with the unpacking operator *, like so:

>>> pairs = [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]
>>> numbers, letters = zip(*pairs)
>>> numbers
(1, 2, 3, 4)
>>> letters
('a', 'b', 'c', 'd')


sample_docs = [100, 5817, 7626, 7356, 1008, 7155, 3533, 13010]
sample_data = [(test_reviews[index],
                test_sentiments[index])
                  for index in sample_docs]


'''

# Separate out list of reviews and sentiments as two different lists from sample_data tuple lists
review_list1 ,sentiment_list1 = zip(*sample_data)

 
# Finding the score of review 
for review, sentiment in zip(review_list1 ,sentiment_list1):
    print('REVIEW:', review)   
    print('Actual Sentiment:', sentiment)   
    print('Predicted Sentiment polarity:', afn.score(review))   
    print('-'*60) 

'''
We can compare the actual sentiment label for each review and check out the predicted sentiment polarity 
score. 
A negative polarity typically denotes negative sentiment. 

To predict sentiment on our complete test dataset of 15,000 reviews 
(I used the raw text documents because AFINN takes into account other aspects like emoticons and 
exclamations), we can now use the following snippet. 
I used a threshold of >= 1.0 to determine if the overall sentiment is positive. 
You can choose your own threshold based on analyzing your own corpora.

'''
    
# Getting the 
sentiment_polarity = [afn.score(review) for review in test_reviews]    

predicted_sentiments = ['positive' if score >= 1.0 else 'negative' for score in sentiment_polarity]

'''
Now that we have our predicted sentiment labels, we can evaluate our model performance based on standard 
performance metrics using our utility function.
'''
import model_evaluation_utils as meu

meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=predicted_sentiments, classes=['positive', 'negative'])


'''
Model Performance metrics:
------------------------------
Accuracy: 0.7118
Precision: 0.7289
Recall: 0.7118
F1 Score: 0.7062

Model Classification report:
------------------------------
             precision    recall  f1-score   support

   positive       0.67      0.85      0.75      7510
   negative       0.79      0.57      0.67      7490

avg / total       0.73      0.71      0.71     15000


Prediction Confusion Matrix:
------------------------------
                 Predicted:         
                   positive negative
Actual: positive       6376     1134
        negative       3189     4301
        
        
I get an overall F1-score of 71%, which is quite decent considering it’s an unsupervised model. 
Looking at the confusion matrix, 
we can clearly see that 
quite a number of negative sentiment-based reviews have been misclassified as positive (3,189) 
and this leads to the lower recall of 57% for the negative sentiment class. 

Performance for the positive class is better with regard to recall or hit-rate, 
where we correctly predicted 6,376 out of 7,510 positive reviews, 
but the precision is 67% because of the many wrong positive predictions made in case of 
the negative sentiment reviews        


#-----------------------------------------------------------------------------------------------------

This blog demonstrates how to evaluate the performance of a model via 
Accuracy, Precision, Recall & F1 Score metrics in Azure ML and 
provides a brief explanation of the “Confusion Metrics”. 
In this experiment, I have used Two-class Boosted Decision Tree Algorithm and my goal is to predict the 
survival of the passengers on the Titanic.

Once you have built your model, the most important question that arises is how good is your model? 
So, evaluating your model is the most important task in the data science project which delineates 
how good your predictions are.

The following figure shows the results of the model that I built for the project I worked on during my internship 
program at Exsilio Consulting this summer.

Accuracy, Precision, Recall & F1 Score

Fig. Evaluation results for classification model

Let’s dig deep into all the parameters shown in the figure above.

The first thing you will see here is ROC curve and we can determine whether our ROC curve is good or 
not by looking at AUC (Area Under the Curve) and other parameters 
which are also called as Confusion Metrics. 

A confusion matrix is a table that is often used to describe the performance of a classification model on 
a set of test data for which the true values are known. 



All the measures except AUC can be calculated by using left most four parameters. 
So, let’s talk about those four parameters first.

Accuracy, Precision, Recall & F1 Score

True positive and true negatives are the observations that are correctly predicted and therefore shown in green. We want to minimize false positives and false negatives so they are shown in red color. These terms are a bit confusing. So let’s take each term one by one and understand it fully.

True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes. E.g. if actual class value indicates that this passenger survived and predicted class tells you the same thing.

True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no. E.g. if actual class says this passenger did not survive and predicted class tells you the same thing.

False positives and false negatives, these values occur when your actual class contradicts with the predicted class.

False Positives (FP) – When actual class is no and predicted class is yes. E.g. if actual class says this passenger did not survive but predicted class tells you that this passenger will survive.

False Negatives (FN) – When actual class is yes but predicted class in no. E.g. if actual class value indicates that this passenger survived and predicted class tells you that passenger will die.

Once you understand these four parameters then we can calculate 
Accuracy, Precision, Recall and F1 score.

Accuracy - Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations. 
One may think that, if we have high accuracy then our model is best. 
Yes, accuracy is a great measure but only when you have symmetric datasets where values of false positive 
and false negatives are almost same. 
Therefore, you have to look at other parameters to evaluate the performance of your model. 
For our model, we have got 0.803 which means our model is approx. 80% accurate.

Accuracy = TP+TN/TP+FP+FN+TN

Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate. We have got 0.788 precision which is pretty good.

Precision = TP/TP+FP

Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. The question recall answers is: Of all the passengers that truly survived, how many did we label? We have got recall of 0.631 which is good for this model as it’s above 0.5.

Recall = TP/TP+FN

F1 score - F1 Score is the weighted average of Precision and Recall. 
Therefore, this score takes both false positives and false negatives into account. 
Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, 
especially if you have an uneven class distribution. 
Accuracy works best if false positives and false negatives have similar cost. 
If the cost of false positives and false negatives are very different, it’s better to look at both 
Precision and Recall. In our case, F1 score is 0.701.

F1 Score = 2*(Recall * Precision) / (Recall + Precision)




#-----------------------------------------------------------------------------------------------
'''        




'''

Unsupervised Sentiment Analysis using SentiWordnet which is based on WordNet Lexicon
This is in nltk Corpus itself

'''


import nltk
from nltk.corpus import sentiwordnet as swn

good = list(swn.senti_synsets('good', 'n'))[0]
print ('Positive Polarity Score:', good.pos_score())
print ('Negative Polarity Score:', good.neg_score())
print ('Objective Score:', good.obj_score())


from normalisation import  normalize_corpus, expand_contractions, strip_html_tags, normalize_accented_characters, tokenize_text, pos_tag_text, remove_special_characters

from contractions import CONTRACTION_MAP

'''

SentiWordNet:
-------------------------------------------------------------------------------------------    
SentiWordNet is a lexical resource for opinion mining. 
SentiWordNet assigns to each synset of WordNet three sentiment scores: positivity, negativity, objectivity. 

SentiWordNet is described in details in the papers:

SentiWordNet: A Publicly Available Lexical Resource for Opinion Mining (view citations)

SentiWordNet 3.0: An Enhanced Lexical Resource for Sentiment Analysis and Opinion Mining (view citations)

The current version of SentiWordNet is 3.0, which is based on WordNet 3.0

https://wordnet.princeton.edu/

WordNet® is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of 
cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of conceptual-semantic 
and lexical relations. The resulting network of meaningfully related words and concepts can be navigated with the 
browser. WordNet is also freely and publicly available for download. WordNet's structure makes it a useful tool 
for computational linguistics and natural language processing.

WordNet superficially resembles a thesaurus, in that it groups words together based on their meanings. However, 
there are some important distinctions. First, WordNet interlinks not just word forms—strings of letters—but specific 
senses of words. As a result, words that are found in close proximity to one another in the network are semantically disambiguated. 
Second, WordNet labels the semantic relations among words, whereas the groupings of words in a thesaurus does not follow any explicit 
pattern other than meaning similarity.


Following function if wordnet gives empty list if word is not present in wordnet
   swn.senti_synsets(word, 'n')
To fetch the sentiment of word using above function, we need to check the following
         # Take the list in some variable as returned by function   
         iteration_set =  list(swn.senti_synsets(word, 'n'))
         # Then check if list has some element in it by checking its length
         if len(iteration_set) > 0 :
                ss_set = list(swn.senti_synsets(word, 'n'))[0]    

Also complete review needs to get cleaned using following functions, so that POS should work correctly
    # Remove all html tags
    review = strip_html_tags(review)    
    # Expand the contractions
    review = expand_contractions(review, CONTRACTION_MAP)
    # make all the words to lower case
    review = review.lower()
    # Remove all puncuation characters
    review = remove_special_characters(review)

  
   
'''
    
def analyze_sentiment_sentiwordnet_lexicon(review,
                                           verbose=False):
    # pre-process text
    review = strip_html_tags(review)    
    review = expand_contractions(review, CONTRACTION_MAP)
    review = review.lower()
    review = remove_special_characters(review)
  
    # tokenize and POS tag text tokens
    text_tokens = nltk.word_tokenize(review)
    
    tagged_text = nltk.pos_tag(text_tokens)
    
    word_list, tag_list = zip(*tagged_text)

    pos_score = neg_score = token_count = obj_score = total_count = 0
    # get wordnet synsets based on POS tags
    # get sentiment scores if synsets are found
    print("Lenght of words: ", len(word_list))
    for word, tag in zip(word_list, tag_list):

        ss_set = None
#        if total_count <= len(word_list)-1:
        if 'NN' in tag:
            print(word)
            iteration_set =  list(swn.senti_synsets(word, 'n'))
            if len(iteration_set) > 0 :
                ss_set = list(swn.senti_synsets(word, 'n'))[0] 
        elif 'VB' in tag:
            iteration_set =  list(swn.senti_synsets(word, 'v'))
            if len(iteration_set) > 0 :
                 ss_set =list(swn.senti_synsets(word, 'v'))[0]
        elif 'JJ' in tag:
            iteration_set =  list(swn.senti_synsets(word, 'a'))
            if len(iteration_set) > 0 :
                 ss_set =list(swn.senti_synsets(word, 'a'))[0]
        elif 'RB' in tag:
            iteration_set =  list(swn.senti_synsets(word, 'r'))
            if len(iteration_set) > 0 :
                # ss_set = list(swn.senti_synsets(word, 'v'))[0]
                 ss_set =list(swn.senti_synsets(word, 'r'))[0]
        # if senti-synset is found        
        if ss_set:
            # add scores for all found synsets
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()
            obj_score += ss_set.obj_score()
            token_count += 1
        total_count += 1
        print("Total Word count counter: ",total_count )
                
    # aggregate final scores
    final_score = pos_score - neg_score
    norm_final_score = round(float(final_score) / token_count, 2)
    final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'


    if verbose:
        norm_obj_score = round(float(obj_score) / token_count, 2)
        norm_pos_score = round(float(pos_score) / token_count, 2)
        norm_neg_score = round(float(neg_score) / token_count, 2)
        # to display results in a nice table
        sentiment_frame = pd.DataFrame([[final_sentiment, norm_obj_score,
                                         norm_pos_score, norm_neg_score,
                                         norm_final_score]],
                                         columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                                                      ['Predicted Sentiment', 'Objectivity',
                                                                       'Positive', 'Negative', 'Overall']], 
                                                              labels=[[0,0,0,0,0],[0,1,2,3,4]]))
        print (sentiment_frame)
        
    return final_sentiment

            
# Test the program for calculating sentiment using  analyze_sentiment_sentiwordnet_lexicon function as defined above           
for review, review_sentiment in sample_data:  
    print ('Review:')
    print ('Labeled Sentiment:', review_sentiment)    
    final_sentiment = analyze_sentiment_sentiwordnet_lexicon(review, verbose=True)
    print ('-'*60)                                                         

# Do predictions for list of 15000 reviews in test_review list
sentiwordnet_predictions = [analyze_sentiment_sentiwordnet_lexicon(review)
                            for review in test_reviews]


from utils import display_evaluation_metrics, display_confusion_matrix, display_classification_report

# Display the Performance Metrices
print ('Performance metrics:')
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=sentiwordnet_predictions,
                           positive_class='positive')  
print ('\nConfusion Matrix:')                           
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=sentiwordnet_predictions,
                         classes=['positive', 'negative'])
print ('\nClassification report:')                         
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=sentiwordnet_predictions,
                              classes=['positive', 'negative'])  

'''
Performance metrics:
--------------------------------
Accuracy: 0.63
Precision: 0.58
Recall: 0.89
F1 Score: 0.7

Confusion Matrix:
----------------------------------    
                 Predicted:         
                   positive negative
Actual: positive       6667      843
        negative       4772     2718

Classification report:
---------------------------------------------------------    
             precision    recall  f1-score   support

   positive       0.58      0.89      0.70      7510
   negative       0.76      0.36      0.49      7490

avg / total       0.67      0.63      0.60     15000

'''

                                                


'''

Unsupervised learning using Vader Lexicon
Already installed with nltk but addtionally following library needs to be installed for twitter analysis

pip install twython

'''

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment_vader_lexicon(review, 
                                    threshold=0.1,
                                    verbose=False):
    # pre-process text
    #review = normalize_accented_characters(review)
    #review = html_parser.unescape(review)
    #review = strip_html(review)
    
    review = strip_html_tags(review)    
    review = expand_contractions(review, CONTRACTION_MAP)
    review = review.lower()
    review = remove_special_characters(review)
  
    
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)
    # get aggregate scores and final sentiment
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold\
                                   else 'negative'
    if verbose:
        # display detailed sentiment statistics
        positive = str(round(scores['pos'], 2)*100)+'%'
        final = round(agg_score, 2)
        negative = str(round(scores['neg'], 2)*100)+'%'
        neutral = str(round(scores['neu'], 2)*100)+'%'
        sentiment_frame = pd.DataFrame([[final_sentiment, final, positive,
                                        negative, neutral]],
                                        columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                                                      ['Predicted Sentiment', 'Polarity Score',
                                                                       'Positive', 'Negative',
                                                                       'Neutral']], 
                                                              labels=[[0,0,0,0,0],[0,1,2,3,4]]))
        print (sentiment_frame)
    
    return final_sentiment
        
    
    

for review, review_sentiment in sample_data:
    print ('Review:')
    print (review)
    print
    print ('Labeled Sentiment:', review_sentiment)    
    print    
    final_sentiment = analyze_sentiment_vader_lexicon(review,
                                                        threshold=0.1,
                                                        verbose=True)
    print ('-'*60)                                                       

vader_predictions = [analyze_sentiment_vader_lexicon(review, threshold=0.1)
                     for review in test_reviews] 

print ('Performance metrics:')
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=vader_predictions,
                           positive_class='positive')  
print ('\nConfusion Matrix:')                           
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=vader_predictions,
                         classes=['positive', 'negative'])
print ('\nClassification report:')                         
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=vader_predictions,
                              classes=['positive', 'negative']) 

'''
Performance metrics:
-----------------------------
Accuracy: 0.7
Precision: 0.66
Recall: 0.85
F1 Score: 0.74

Confusion Matrix:
                 Predicted:         
                   positive negative
Actual: positive       6385     1125
        negative       3352     4138

Classification report:
             precision    recall  f1-score   support

   positive       0.66      0.85      0.74      7510
   negative       0.79      0.55      0.65      7490

avg / total       0.72      0.70      0.69     15000

'''  
