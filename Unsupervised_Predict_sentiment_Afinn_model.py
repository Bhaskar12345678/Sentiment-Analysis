# -*- coding: utf-8 -*-
"""
Created on Tue Feb 01 19:45:40 2020

@author: Bhaskar Mahna


Description: Unsupervised Machine Learning using various Lexicon

 Using Afinn Lexicon dictionary library to score sentiment for movie review passed as parameter
 
 
Install Afinn library using
pip install Afinn in virtual env.

"""

from afinn import Afinn
import sys
from bs4 import BeautifulSoup

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def Unsupervised_Predict_sentiment_Afinn_model(review):
    
    afn = Afinn(emoticons=True) 
    doc = strip_html_tags(review)
       
    '''
    #doc = doc.lower().strip()
    '''
        
    sentiment_polarity = afn.score(doc)    
    
    #print('{"sentiment":' + sentiment_polarity + '}')
    
    if (sentiment_polarity >= 1.0):
        print('{"sentiment":' + '"positive"'+ '}')
    else:
        print('{"sentiment":' + '"negative"'+ '}')
    

if __name__ == '__main__':
# Map command line arguments to function arguments.
    Unsupervised_Predict_sentiment_Afinn_model(*sys.argv[1:])