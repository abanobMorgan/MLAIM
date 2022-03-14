

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import cleaningData
import pickle


def modelPreporcess(df,balance_val=0 ):
  """
  input:
      df: dataframe pandas that hold data records 
      balance_val: number of recored the will be in each targets. 
    output 6 values: 
      X: features of the balanced dataframe 
      y: targets of the balanced dataframe 
      
    this function talks df, balance_val and return x: for features, y: for targets of the blanced data
      
  """
  balance_val = min(df.dialect.value_counts())
  #balance_val=7000
  df, _ = cleaningData.balance_data(df, balance_val )


  X = df['text']
  y = df['dialect']
  return X, y
def model_Split(X,y,test_size=0.4 ):
  """
    input:
      X: text pandas series the hold the features of the data
      y: the data target series
    output 6 values: 
      X_train: the train data features 
      X_val: the val  data features 
      X_test: the test data features  
      y_train: the train targets 
      y_val: the val targets 
      y_test: the test targets 
      
    this function talks x, y and return the train data with is 60% of the data and 
    20% for validation and 20% for test as defult values 
  """
  
  # split the data into train remain 60 to 40 --> split the remain into val and test 
  # using stratify=y to have the same presantage of the target in the train, val and test.
  X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
  X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state=0, stratify=y_remain)

  return X_train, X_val, X_test, y_train, y_val, y_test 

def data_fit_transform(X_train,X_val,X_test): 
  """
  inputs: 
    X_train: the features that will be used for fittin the tf-idf transformation, then transform it  
    X_val: validation features that will be transformed with the same tarnformation for train 
    X_test: testing features that will be transformed with the same tarnformation for train 
  output: 
    X_train:  transformed data   
    X_val: transformed data 
    X_test: transformed data 
    vectr: the tf-idf verctorizer. 

  this function fit then transform the data into tf-idf vectoizer 
   
  """ 
  # create  term frequency–inverse document frequency

  vectr = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=10, max_df= .8  )
  vectr.fit(X_train)
  val_X = vectr.transform(X_val)
  test_X = vectr.transform(X_test)
  train_X = vectr.transform(X_train)

  return train_X, val_X, test_X, vectr

def data_fit(X, vec): 
  """
  inputs: 
    X: the features that will be used for predicting 
    vec: vectorizer that will be use to transform the data 
  output: 
    transformed data 

  this function transform the data into tf-idf vectoizer 

  """ 
  return vec.transform(X)

def trainModel(train_X,y_train): 
  """
  inputs: 
    train_X: the features that will be used for predicting 
    y_train: targets of the data
  output: 
    model that used to fit the data 

  this function fit the train data into a model. 
  """ 

  model = LogisticRegression(multi_class='multinomial')
                                  
  clf=model.fit(train_X,y_train)

  return clf 

def testModel(model , X, y): 
  """
  inputs: 
    model: model that will be used for prediction 
    X: the features that will be used for predicting 
    y: targets of the data
  output: 
    the data score for the predicted data 

  this function calculate the score of the prediction. 
  """ 
  return model.score(X,y)*100

def predictData(model, X): 
  """
  inputs: 
    model: model that will be used for prediction 
    X: the features that will be used for predicting 
  output: 
    predicted data 

  this function predict the data using  a model, and feature(X) 
  """ 
  return model.predict(X)

def saveModel(filePath, model):
  """
  inputs: 
    model: model to be saved in pickle dumps 
    filepath: the path this model will be saved in 
  output: 
    this function has no output it just save data

  this function save the model file 
  """ 
  pickle.dump(model, open(filePath, 'wb'))
  print(f"model saved in {filePath}")

def saveVectorizer(vectorizer, filePath): 
  """
  inputs: 
    vectorizer: tf-idf vectorizer
    filepath: the path this vectorizer will be saved in 
  output: 
    this function has no output it just save data

  this function save the tf-idf verctorizer into pk file 
  """
  with open(filePath, 'wb') as fin:
    pickle.dump(vectorizer, fin)
  
  print(f"model saved in {filePath}")

"""
test_z = vectr.transform([" احنا بيقنا الصبح استاذ مجدي يومك بيضحك"])
print(model.predict(test_z))"""
