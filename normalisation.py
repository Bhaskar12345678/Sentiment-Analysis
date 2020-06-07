# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 04:28:07 2020

@author: Bhaskar Mahna

Description: Library of NLP processing function to pre-process the 
             text before inputed for doing either training or prediction of text reviews in various programs
             List of functions includes:
                 - Stripping unwanted HTML tags from the text - strip_html_tags
                 - Tokenizer - tokenize_text
                 - Punctuation remover - remove_punctuations
                 - Expand Contractions like isn't to is not etc. - expand_contractions
                 - Part of speech tagging - pos_tag_text
                 - Lemmatize text - lemmatize_text 
                 - remove_special_characters
                 - remove_stop_words
                 - keep_text_characters
                 - normalize_accented_characters
                 - remove_repeated_characters
                 - normalize_corpus 
"""

import nltk
#nltk.download()

from contractions import CONTRACTION_MAP
import re
from nltk.stem import WordNetLemmatizer
#from HTMLParser import HTMLParser
import unicodedata

import string
from bs4 import BeautifulSoup 

stopword_list = nltk.corpus.stopwords.words('english')
stopword_list = stopword_list + ['mr', 'mrs', 'come', 'go', 'get',
                                 'tell', 'listen', 'one', 'two', 'three',
                                 'four', 'five', 'six', 'seven', 'eight',
                                 'nine', 'zero', 'join', 'find', 'make',
                                 'say', 'ask', 'tell', 'see', 'try', 'back',
                                 'also']
wnl = WordNetLemmatizer()



def strip_html_tags(text):    
    soup = BeautifulSoup(text, "html.parser")    
    [s.extract() for s in soup(['iframe', 'script'])]    
    stripped_text = soup.get_text()    
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    
    return stripped_text


# Word Tokenizer from nltk
def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
  
    #  Following is already done in removeSpecialCharacter function below - pls check
    # remove all punctuation characters from text !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    # tokens = [remove_punctuations(token) for token in tokens]
    
    return tokens

def remove_punctuations(text):

    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # traverse the given string and if any punctuation 
    # marks occur replace it with null 
    for x in text.lower(): 
        if x in punctuations: 
            text = text.replace(x, "") 
  
    return text


'''
Contractions are shortened versions of words or syllables. These exist in written and spoken forms. 
Shortened versions of existing words are created by removing specific letters and sounds. In the case 
of English contractions, they are often created by removing one of the vowels from the word. 
Examples include “is not” to “isn’t” and “will not” to “won’t”, 
where you can notice the apostrophe being used to denote the contraction and some of the vowels and other letters being removed. 


By nature, contractions pose a problem for NLP and text analytics because, to start with, we have a 
special apostrophe character in the word. Besides this, we also have two or more words represented by a 
contraction and this opens a whole new can of worms when we try to tokenize them or standardize the words. 
Hence, there should be some definite process for dealing with contractions when processing text. 

'''
def expand_contractions(text, contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
    
    


import nltk
from nltk.corpus import wordnet as wn

from nltk.parse import CoreNLPParser



# POS Tagger

#tagged_lower_text = pos_tag_text("What is the airspeed of an unladen swallow ?")

# Annotate text tokens with POS tags from CoreNLP pos tagger
def pos_tag_text(text):
    
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    #print("ORIGINAL TEXT ---------------",text)
    #tagged_text = tag(text)
    pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
    tagged_text = pos_tagger.tag(text.split())

    
    #tagged_text = nltk.pos_tag(text)

    #print("TAGGED TEXT ---------",tagged_text)

     
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    
    
    return tagged_lower_text
    
# lemmatize text based on POS tags    
def lemmatize_text(text):
    
    pos_tagged_text = pos_tag_text(text)
    #print(pos_tagged_text)

    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word                     
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text
    

# Remove Punctuation etc.
#  Removing Special Characters
# Special characters and symbols are usually non-alphanumeric characters or even occasionally numeric characters (depending on the problem), 
# which add to the extra noise in unstructured text    
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
    
# Remove Stopwords to reduce noise    
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


# Keep only Text characters as numbers will not help in sentiment analysis
def keep_text_characters(text):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text



'''
Removing Accented Characters
Usually in any text corpus, we will be dealing with accented characters/letters, 
especially if you only want to analyze the English language. Hence, we need to make 
sure that these characters are converted and standardized into ASCII characters. 
This shows a simple example — converting é to e. 
The following function is a simple way of tackling this task.
'''


def normalize_accented_characters(text):
    #print(text)
    #text = unicodedata.normalize('NFKD',
    #                                     text.decode('utf-8', 'replace')
    #                                     ).encode('ascii', 'ignore')
    text = text.decode('utf-8', 'ignore')
    text = unicodedata.normalize('NFKD', text.encode('ascii', 'ignore'))
     
    return text



from nltk.corpus import wordnet 
def remove_repeated_characters(tokens):
    
    old_word = 'finalllyyy' 
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)') 
    match_substitution = r'\1\2\3' 
    step = 1
    while True:    # check for semantically correct word    
        if wordnet.synsets(old_word):        
            #print("Final correct word:", old_word)        
            break
        # remove one repeated character    
        new_word = repeat_pattern.sub(match_substitution, old_word)    
        if new_word != old_word:        
            #print('Step: {} Word: {}'.format(step, new_word))        
            step += 1 # update step        
            # update old word to last substituted state        
            old_word = new_word        
            continue    
        else:        
            #print("Final word:", new_word)        
            break
    return new_word    

# Main function calling all other functions to normalize the text corpus
def normalize_corpus(corpus, lemmatize=True, 
                     only_text_chars=False,
                     tokenize=False):
    
    normalized_corpus = []    
    for index, text in enumerate(corpus):
        #text = normalize_accented_characters(text)
        #text = html_parser.unescape(text)
        #text = strip_html(text)
        text = strip_html_tags(text)    
        text = expand_contractions(text, CONTRACTION_MAP)
        # print(text)
        if lemmatize:
            text = lemmatize_text(text)
        else:
            text = text.lower()
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        if only_text_chars:
            text = keep_text_characters(text)
        
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)
            
    return normalized_corpus

# Parsing for sentence tokenizing
def parse_document(document):
    document = re.sub('\n', ' ', document)
    if isinstance(document, str):
        document = document
    elif isinstance(document, unicode):
        return unicodedata.normalize('NFKD', document).encode('ascii', 'ignore')
    else:
        raise ValueError('Document is not string or unicode!')
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences
    
    