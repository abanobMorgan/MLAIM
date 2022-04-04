import datafetching
import clean_preprocess_main
import LogisticRegression 
import pandas as pd 
import BILSTM

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from gensim.models import word2vec
import word2vector
import pickle

def fetech_clean_preprocess():
  """
  this function call some fuction for fetch the data and clean it then preprocess it 
  lastly save it into a file
  
  """
  datafetching.fetch()
  clean_preprocess_main.run()

def ML(): 
    """
    thif function works as run for the machine learing model
    """
  
    df = pd.read_csv("../data/balance_data.txt")
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
    LogisticRegression.saveVectorizer(vectorizer, vectorizerpath)    

def DL(): 
  WORDSMODEL ="../model/word2vec_twitter_Model.model"
  DATAPATH = "../data/balance_data.txt"
  MODELPATH= "../model/lstm"
  
  try:
    modle = pickle.load(MODELPATH, 'rb')
  except:  
    df = pd.read_csv(DATAPATH)
    df.dropna(inplace=True)
    NUMWORDS= 20000
    max_length = 100
    EPOCH = 30
    CLASS_NAMES =list(df.dialect.unique())
    NUMBER_OF_CLASSES = len(CLASS_NAMES)
    trunc_type = 'post' 
    padding_type = 'post'
    OOV_tok = '<OoV>'

    X, y  = LogisticRegression.modelPreporcess(df)
    X_train,X_val, X_test,y_train, y_val, y_test = LogisticRegression.model_Split(X, y)
   
    tokenizer= BILSTM.tokenizer_fit(X_train,NUMWORDS)
    embedding_matrix= word2vector.get_embedding(tokenizer= tokenizer)

    train_padded=  BILSTM.tokenizer_transform(X_train, tokenizer, max_length,padding_type )
    val_padded=BILSTM.tokenizer_transform(X_val, tokenizer, max_length,padding_type )
    test_padded=BILSTM.tokenizer_transform(X_test, tokenizer, max_length,padding_type )

    label_tokenizer=  BILSTM.label_tokeniz_fit(y)

    train_labels_seq=BILSTM.label_tokeniz_transform(y_train,label_tokenizer )
    val_labels_seq = BILSTM.label_tokeniz_transform(y_val,label_tokenizer )
    test_labels_seq = BILSTM.label_tokeniz_transform(y_test,label_tokenizer )

    vocab_size = len(tokenizer.word_index) + 1 

    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, cooldown=0),
             EarlyStopping(monitor='val_accuracy', min_delta=1e-7, patience=8)]
    model = BILSTM.create_model(vocab_size, embedding_matrix.shape[1], max_length, embedding_matrix)
    #BILSTM.plotModel(model)
    #print('model ploted')
    history = model.fit(x= train_padded,y= train_labels_seq,
                         callbacks=callbacks,
                      validation_data=(val_padded,val_labels_seq),
                      epochs=EPOCH,
                      batch_size=624
                      )
    print("done")


    LogisticRegression.saveModel(MODELPATH, model)
"""
if(__name__ == "__main__"):
    fetech_clean_preprocess()
    ML()""" 
DL()






