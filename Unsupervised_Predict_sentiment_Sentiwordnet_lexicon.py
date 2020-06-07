# -*- coding: utf-8 -*-
"""
Created on Feb 28  2 13:21:07 2020

@author: Bhaskar Mahna


"""

import sys
import nltk
from nltk.corpus import sentiwordnet as swn
import pandas as pd

from normalisation import  normalize_corpus, expand_contractions, strip_html_tags, normalize_accented_characters, tokenize_text, pos_tag_text, remove_special_characters

from contractions import CONTRACTION_MAP


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
    #print("Lenght of words: ", len(word_list))
    for word, tag in zip(word_list, tag_list):

        ss_set = None
#        if total_count <= len(word_list)-1:
        if 'NN' in tag:
            # print(word)
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
        # print("Total Word count counter: ",total_count )
                
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
        # print (sentiment_frame)
        
    return final_sentiment



def Unsupervised_Predict_sentiment_Sentiwordnet_lexicon(review):
    
    
    final_sentiment =  analyze_sentiment_sentiwordnet_lexicon(review, verbose=False)        
    
    print('{"sentiment": "' + final_sentiment + '"}')

if __name__ == '__main__':
# Map command line arguments to function arguments.
    Unsupervised_Predict_sentiment_Sentiwordnet_lexicon(*sys.argv[1:])
    
    