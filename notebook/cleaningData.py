import pandas as pd
import re
from nltk.corpus import stopwords





def getData(text):
    """
    input: 
        this function has no input. 
    outputs: 
        dict data. 

    this function read the data from a data folder adn retrun a dict that have the data records
    
    """
    file = open(f'../data/{text}.txt', 'r', encoding="utf-8")
    data = file.readlines()
    myDic = []
    for line in data: 
        if line != "\n":
            myDic.append(line.split(','))
                 
    file.close()
    return myDic

def removing_http(text):
    """
    input: 
        text: string  
    outputs: 
        string of  data without https links. 
        
    this function get a text and return cleaned one.     
    """
    pattern=r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*'
    clean_text = re.sub(pattern, " ", text)
    return clean_text

def removing_tag(text):
    """
    input: 
        text: string  
    outputs: 
        arabic string without mentions and numbers. 
        
    this function get removing the tags from the tweets.  
    """
    #removing the tags from the tweets 
    pattern=r'([@A-Za-z0-9_]+)|r"\(\[\[A-Za-z0-9_]+]*\)"'
    clean_text = re.sub(pattern, " ", text)
    
    return clean_text
def clean(text): 
    """
    input: 
        text: string  
    outputs: 
        arabic string only. 
        
    this function  remove any thing is not in arabic language then remove the white spaces.  
    """
    #remove any thing is not in arabic language then remove the white spaces 
    pattern=r'\W'
    clean_text = re.sub(pattern, " ", text, flags=re.UNICODE)
    clean_text = re.sub('\s+', " ", clean_text, flags=re.UNICODE)
    return clean_text

def eliminate_single_char_words(text):
    """
    input: 
        text: string  
    outputs: 
        arabic string only. 
        
    this function  remove word has len equals to one  
    """
    parts = text.split(" ")
    cleaned_line_parts = []
    for P in parts:
        if len(P) != 1:
            cleaned_line_parts.append(P)
    cleaned_line = ' '.join(cleaned_line_parts)
    return cleaned_line

def ReplaceThreeOrMore(text):
    """
    input: 
        text: string  
    outputs: 
        arabic string only. 
        
    this function  remove any word that has three or more repetitions of any character, including 
    """
    # pattern to look for three or more repetitions of any character, including
    # newlines.
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    return pattern.sub(r"\1\1", text)

def mergeData(df1): 
    """
    input: 
        df1: pandas dataframe 
            
    this function  get the data into a df format and merge it with the targets to be used a one
    file. Then save this into a file in the data called margedData.txt

    """
    df2 = pd.read_csv("../doc/dialect_dataset.csv")
    df1.id=df1["id"].astype(float).astype('Int64')
    df= pd.merge(df1, df2, on="id")
    df.drop(columns='id', inplace=True)
    df.to_csv('../data/mergedData.txt', encoding='UTF-8' , index=False )

def balance_data(df, balance_val):
    """
    input: 
        df: pandas dataframe 
        balance_val: int value   
    outputs: 
        balance_df: data frame that hold teh data from the first element to the balance value
        remain_df: the remain of the data 
        
    this function  split the data into two data frame one have the first X values and the 
    remain of the data save into the remain_df. 
    this X is  balance_val user difined or can be min of the target value_counts   
    """
  
    balance_df= pd.DataFrame(columns=['text', 'dialect'])
    remain_df = pd.DataFrame(columns=['text', 'dialect'])
    for i in df.dialect.unique():
        new = df[df.dialect == i]
        row_df = pd.DataFrame(new)
    
        balance_df = pd.concat([row_df.iloc[:balance_val], balance_df], ignore_index=True)
        remain_df = pd.concat([row_df.iloc[balance_val:], balance_df], ignore_index=True)
    print(balance_df.shape)
    balance_df.to_csv('../data/balance_data.txt', encoding='UTF-8' , index=False )
    remain_df.to_csv('../data/remain_data.txt', encoding='UTF-8' , index=False )
    return balance_df, remain_df