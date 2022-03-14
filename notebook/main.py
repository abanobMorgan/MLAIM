import datafetching
import clean_preprocess_main
import LogisticRegression 
import pandas as pd 

def ML(): 
    """
    thif function works as run for the machine learing model
    """
    datafetching.fetch()
    clean_preprocess_main.run()

    """df = pd.read_csv("../data/balance_data.txt")
    df.dropna(inplace=True) 
    
    X, y = LogisticRegression.modelPreporcess(df)

    X_train, X_val, X_test, y_train, y_val, y_test= LogisticRegression.model_Split(X,y)

    X_train,X_val,X_test, vectorizer= LogisticRegression.data_fit_transform(X_train,X_val,X_test)

    model = LogisticRegression.trainModel(X_train,y_train)

    scores= LogisticRegression.testModel(model , X_val, y_val)
    print(scores)
    
    modelPath= "../model/LogisticRegression1"
    vectorizerpath='../model/vectorizer1.pk'
    LogisticRegression.saveModel(modelPath, model)
    LogisticRegression.saveVectorizer(vectorizer, vectorizerpath)   """ 


if(__name__ == "__main__"):
     ML() 






