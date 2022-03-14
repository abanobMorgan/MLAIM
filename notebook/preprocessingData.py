
import re 
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import wordpunct_tokenize
import tashaphyne.arabic_const as arabconst
import pandas as pd 
import pickle   

def remove_stopword(text, stopwords_list):  
    """
    input: 
        text: string  
        stopwords_list: pandas series that have all arabic stop words 
    outputs: 
        arabic string without stop words. 
        
    this function  remove any stop word from the  arabic text.  
    """
    stopwords_list = pd.Series(stopwords_list).array
    a=[w for w in wordpunct_tokenize(text) if w not in stopwords_list]
    return ' '.join(a)


    
def strip_tatweel(text):
    """
    input: 
        text: string  
    outputs: 
        string. 
        
    this function removing ـــــــ from an arabic word.  
    """
    #removing ـــــــ from an arabic word
    return re.sub(r'[%s]' % arabconst.TATWEEL,    '', text)
def strip_tashkeel(text): 
    """
    input: 
        text: string  
    outputs: 
        string. 
        
    this function removing التشكيل from the data.
    """
    # removing التشكيل from the data 
    return arabconst.HARAKAT_PAT.sub('', text)
def normalize_hamza(text):
    """
    input: 
        text: string  
    outputs: 
        string. 
        
    this function handiling الهمزه.
    """
    # handiling الهمزه 
    text = arabconst.ALEFAT_PAT.sub(arabconst.ALEF, text) 
    return arabconst.HAMZAT_PAT.sub(arabconst.HAMZA, text)
def normalize_lamalef(text):
    """
    input: 
        text: string  
    outputs: 
        string. 
        
    this function handiling لآ.
    """

     # handiling لآ  
    return arabconst.LAMALEFAT_PAT.sub(r'%s%s'%(arabconst.LAM, arabconst.ALEF), text)
def normalize_spellerrors(text):
    """
    input: 
        text: string  
    outputs: 
        string. 
        
    this function handiling التاء المربوطه و الالف المكسوره.
    """
    #handiling التاء المربوطه و الالف المكسوره 
    text = re.sub(r'[%s]' % arabconst.TEH_MARBUTA,    arabconst.HEH, text) 
    return re.sub(r'[%s]' % arabconst.ALEF_MAKSURA,    arabconst.YEH, text)

def normalize_searchtext(text):
    """
    input: 
        text: string  
    outputs: 
        string. 
        
    this function call the strip_tashkeel then strip_tatweel thennormalize_lamalef
    then normalize_hamza then normalize_spellerrors 
    to remove all arabic variation  
    """
    text = strip_tashkeel(text) 
    text = strip_tatweel(text) 
    text = normalize_lamalef(text) 
    text = normalize_hamza(text) 
    text = normalize_spellerrors(text) 
    return text 

def extract_terms(text):
    """
    input: 
        text: string  
    outputs: 
        string. 
    
    this function perform stemming into the text and return the unique words  
    """
    
    tokens = text.split()
    # stemming
    stemmer = ISRIStemmer()

    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))

    return ' '.join(stemmed)

def term_frequency(data):
    """
    input: 
        data: pandas data frame    
        
    this function take a pandas dataframe and write a tfidf vector into a pickle file 
    """
    print(data.shape)
    vectr = TfidfVectorizer(ngram_range=(1,2),min_df=1)
    vectr.fit(data)  
    with open('../model/vectorizer.pk', 'wb') as fin:
        pickle.dump(vectr, fin)
