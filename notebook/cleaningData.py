import numpy as np
import pandas as pd
import re
import tashaphyne.arabic_const as arabconst


def getData():
    file = open('../data/finaloutput.txt', 'r', encoding="utf-8")
    data = file.readlines()
    myDic = []
    for line in data: 
        if line != "\n":
            myDic.append(line.split(','))
                 
    file.close()
    return myDic

def removing_http(text):
    pattern=r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*'
    clean_text = re.sub(pattern, " ", text)
    return clean_text


def clean(text): 
    #remove any thing is not in arabic language then remove the white spaces 
    pattern=r'\W'
    clean_text = re.sub(pattern, " ", text, flags=re.UNICODE)
    clean_text = re.sub('\s+', " ", clean_text, flags=re.UNICODE)
    
    return clean_text


def removing_tag(text):
    #removing the tags from the tweets 
    pattern=r'([@A-Za-z0-9_]+)|r"\(\[\[A-Za-z0-9_]+]*\)"'
    clean_text = re.sub(pattern, " ", text)
    
    return clean_text
    
def strip_tatweel(text):
    #removing ـــــــ from an arabic word
    return re.sub(r'[%s]' % arabconst.TATWEEL,    '', text)
def strip_tashkeel(text): 
    # removing التشكيل from the data 
    return arabconst.HARAKAT_PAT.sub('', text)
def normalize_hamza(text):
    # handiling الهمزه 
    text = arabconst.ALEFAT_PAT.sub(arabconst.ALEF, text) 
    return arabconst.HAMZAT_PAT.sub(arabconst.HAMZA, text)
def normalize_lamalef(text):
     # handiling لآ  
    return arabconst.LAMALEFAT_PAT.sub(r'%s%s'%(arabconst.LAM, arabconst.ALEF), text)
def normalize_spellerrors(text):
    #handiling التاء المربوطه و الالف المكسوره 
    text = re.sub(r'[%s]' % arabconst.TEH_MARBUTA,    arabconst.HEH, text) 
    return re.sub(r'[%s]' % arabconst.ALEF_MAKSURA,    arabconst.YEH, text)

def normalize_searchtext(text):
    text = strip_tashkeel(text) 
    text = strip_tatweel(text) 
    text = normalize_lamalef(text) 
    text = normalize_hamza(text) 
    text = normalize_spellerrors(text) 
    return text 


theNewDictionary = getData()
print(len(theNewDictionary))
df = (pd.DataFrame(theNewDictionary))
df.shape
df.head()

df[1] = df.iloc[:,1].apply(lambda x: removing_http(x))
df[1] = df.iloc[:,1].apply(lambda x: removing_tag(x))
df[1] = df.iloc[:,1].apply(lambda x: clean(x))
df[1] = df.iloc[:,1].apply(lambda x: normalize_searchtext(x))

df.head()
df.to_csv('../data/cleandata.txt', encoding='UTF-8' , header= ['id', 'text'], index=False )
print('done')