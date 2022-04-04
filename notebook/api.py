from copyreg import clear_extension_cache
import pickle
import pandas as pd
from flask import Flask, request, jsonify
import clean_preprocess_main


def predictTextValML(data): 
    """
    input:
      text: string data will be used for predicting. 
    output : 
      result: the label of this text dialect 

          
    this function talks text from the user in website form then return the dialect value as reponse
    this function works as following: 
    reading the wights from pretrained data , then use them to predict the value
      
  """
    loaded_model = pickle.load(open("../model/LogisticRegression", 'rb'))
    victor =  pickle.load(open("../model/vectorizer.pk", 'rb'))

    X_val= victor.transform(data)

    result = loaded_model.predict(X_val)
    print(result)
    return(result)



def predictTextValDeepLearning(data): 
    """
    input:
      text: string data will be used for predicting. 
    output : 
      result: the label of this text dialect 

          
    this function talks text from the user in website form then return the dialect value as reponse
    this function works as following: 
    reading the wights from pretrained data , then use them to predict the value
      
  """
    loaded_model = pickle.load(open("../model/LSTM", 'rb'))
    victor =  pickle.load(open("../model/DLvectorizer.pk", 'rb'))

    
    X_val= victor.transform(data)

    result = loaded_model.predict(X_val)
    print(result)
    return result

def textCleanForModel(text): 
    """
    input:
      text: string data will be used for predicting. 
    output : 
      data: panads series that hold the text data 

          
    this function talks text from the user in website form then return the cleaned text in form of 
    padas series to be used in prediction       
    """

    data = pd.Series(text)
    data = clean_preprocess_main.finalClean(data)
    print(data)
    return data 


# Get request --> send no data 
# Post request --> send data 
# return --> response 
app = Flask(__name__) # app : Flask API 

@app.route("/") # get method 
def hello():
    return "Hello world"  #reponse


  

@app.route("/MLpredict",  methods=["POST"])
def MLpredict():
    if request.method == 'POST':
        text = str(request.form.get("area"))
        data= textCleanForModel(text)
        return str(predictTextValML(data))
                

@app.route("/DLpredict",  methods=["POST"])
def DLpredict():
    if request.method == 'POST':
        text = str(request.form.get("area"))
        data= textCleanForModel(text)
        return str(predictTextValDeepLearning(data))




if __name__ == "__main__":
    
    app.run()