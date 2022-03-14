
from lib2to3.pgen2.pgen import DFAState
import cleaningData
import preprocessingData
import pandas as pd 
from  nltk.corpus import stopwords
import nltk 


def finalClean(df1): 
    '''
    input: 
        df is a pandas data frame
    
    output: 
        df: the clean data frame
    
    this function takes a dataframe and normalize the stopwords, https, tags, any word does not in
        arabic language,  normalize the text, remove the stop wrods, then remove any repeated letter in 
        sequesce more than 2 such as ههههههه ،ازااااي , lastly remove any wrod with length of 1 
        
    '''
    try : 
        stopwords_list = stopwords.words('arabic')
    except: 
        nltk.download("stopwords")

    stopwords_list = pd.Series(stopwords_list)
    stopwords_list= stopwords_list.apply(lambda x: preprocessingData.normalize_searchtext(x))
    df1 = df1.iloc[:].apply(lambda x: cleaningData.removing_http(x))
    print('https removed')
    df1 = df1.iloc[:].apply(lambda x: cleaningData.removing_tag(x))
    print('tags removed')

    df1 = df1.iloc[:].apply(lambda x: cleaningData.clean(x))
    print('the data cleaned')

    df1 = df1.iloc[:].apply(lambda x: preprocessingData.normalize_searchtext(x))
    print('normalization done')


    df1 = df1.iloc[:].apply(lambda x: preprocessingData.remove_stopword(x, stopwords_list))
    print('stop words removed')

    df1 = df1.iloc[:].apply(lambda x: cleaningData.ReplaceThreeOrMore(x))
    df1 = df1.iloc[:].apply(lambda x:  cleaningData.eliminate_single_char_words(x))
    print('removed 3+ and 1 letter words ')

    return df1 

def enhance_the_data(df): 
    '''
    input: 
        df is a pandas data frame
    
    output: 
        has no output
    
    this function takes a dataframe create columns name for it and then save it in cleandata file
    then call mergedata function that merge the target with text and save it into mergeddata file 
    then read the data from merged file and drop nans the balance the data by making each dialect 
    has the same number of rows and save the data. 
    '''   
    df.columns= ['id', 'text']
    df.to_csv('../data/cleandata.txt', encoding='UTF-8' , index=False )
    print('saved')
    cleaningData.mergeData(df)
    print('the data created successfully')
    df = pd.read_csv("../data/mergedData.txt")
    df.dropna(inplace=True)
    balance_val = min(df.dialect.value_counts())
    df, _ = cleaningData.balance_data(df, balance_val)
    print(f"the data balanced with value= {balance_val} and saved them into two df ")

    print('done')

def run(): 
    """
    this function call all the clean preprocess main to create the final data files witch is 
    balance_data.text
    """
    theNewDictionary = cleaningData.getData("finaloutput")
    df = (pd.DataFrame(theNewDictionary))
    print(df.shape)
    df1 = df[1]
    final = finalClean(df1)
    df[1]= final
    enhance_the_data(df)
    

if(__name__ == "__main__"): 
    run()
    