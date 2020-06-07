# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:44:34 2020

@author: Bhaskar Mahna
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys
import pandas as pd

from normalisation import   expand_contractions, strip_html_tags, remove_special_characters

from contractions import CONTRACTION_MAP

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
    final_sentiment = 'positive' if agg_score >= threshold  else 'negative'
  
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
       # print (sentiment_frame)
    
    return final_sentiment
        
    



def Unsupervised_Predict_sentiment_Vader_lexicon(review):
    
    
    final_sentiment = analyze_sentiment_vader_lexicon(review,
                                                        threshold=0.1,
                                                        verbose=False)
    
    print('{"sentiment": "' + final_sentiment + '"}')

if __name__ == '__main__':
# Map command line arguments to function arguments.
    Unsupervised_Predict_sentiment_Vader_lexicon(*sys.argv[1:])
    
     

