# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 08:48:33 2020

@author: Bhaskar Mahna
Description:  Program to predict the input review or text for Postive or Negative Sentiment based on saved model 
              trained on LSTM keras Embedded layer 

review = r''' A wonderful little production. <br /><br />The 
        filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, 
        sense of realism to the entire piece. <br /><br />The actors are extremely well chosen- Michael Sheen not only "has got 
        all the polari" but he has all the voices down pat too! You can truly see the seamless editing guided by the references 
        to Williams' diary entries, not only is it well worth the watching but it is a terrificly written and performed piece. 
        A masterful production about one of the great master's of comedy and his life. <br /><br />The realism 
        really comes home with the little things: the fantasy of the guard which, rather than use the 
        traditional 'dream' techniques remains solid then disappears. It plays on our knowledge and our senses, 
        particularly with the scenes concerning Orton and Halliwell and the sets (particularly of their flat with Halliwell's murals decorating every 
        surface) are terribly well done. '''

    review = '''Bobcat Goldthwait should be commended for attempting to do something different with this surprisingly heartfelt film, a cautionary tale about the pitfalls of being honest about everything. Melinda Hamilton stars as Amy, a girl who has had oral sex with a canine in the past on a lark. She struggles with telling her fiancé, John. Of course the truth does rear it's shaggy ugly head. The film deals with the fallout of said escapade. The movie is well-acted by all, save for perhaps Jack Plotnick as Dougie, who never really felt like he mashed well with the picture. And the film while solid enough seems to miss it's mark a few times. Every single person in the film struggles with massive hypocrisy and all our a tad hard to relate to. Bobcat should be commended for doing something different, as I said before, but different does not always equal good and this pales ever so slightly not to Goldthwaits own directorial debut, the criminally misunderstood "Shakes the Clown'''

review = '''Phil the Alien is one of those quirky films where the humour is based around the oddness of everything rather than actual punchlines.<br /><br />At first it was very odd and pretty funny but as the movie progressed I didn\'t find the jokes or oddness funny anymore.<br /><br />Its a low budget film (thats never a problem in itself), there were some pretty interesting characters, but eventually I just lost interest.<br /><br />I imagine this film would appeal to a stoner who is currently partaking.<br /><br />For something similar but better try "Brother from another planet"'''

review = '''This an excellent movie worth watching'''
review = reviews[101]
review
a[100]


"""
import sys

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from bs4 import BeautifulSoup
import re

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

model_file='C:/Bhaskar Sem 8/Capstone project/Capstome Sem 8 Documents submitted to College/Sentiment_Analysis/saved_models/model.h5'
# Based on trained saved model max length model
padded_max_len=2087

def Predict_sentiment_LSTM_model(review):
    




    clean_docs = []
    
    doc = strip_html_tags(review)
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
    X = sequence.pad_sequences(X,  maxlen=padded_max_len)
    
   
    # Load saved model 
    model = load_model(model_file)
    
    
    
    r = model.predict_classes(X)
    #print(r[0])


    if r[0] == 0:
        print('{"sentiment":' + '"negative"'+ '}')
    else:
        print('{"sentiment":' + '"positive"'+ '}')

       
    
if __name__ == '__main__':
    # Map command line arguments to function arguments.
    Predict_sentiment_LSTM_model(*sys.argv[1:])