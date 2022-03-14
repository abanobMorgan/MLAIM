import datafetching
import clean_preprocess_main
import LogisticRegression 
import pandas as pd 
import BILSTM

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

    
  df = pd.read_csv("../data/balance_data.txt")
  df.dropna(inplace=True)

  #np.mean(df.text.map(str).apply(len))

  max_length = 160
  EPOCH = 30
  CLASS_NAMES =list(df.dialect.unique())
  NUMBER_OF_CLASSES = len(CLASS_NAMES)
  trunc_type = 'post' 
  padding_type = 'post'
  OOV_tok = '<OoV>'

  X, y  = LogisticRegression.modelPreporcess(df)


  X_train,X_val, X_test,y_train, y_val, y_test = LogisticRegression.model_Split(X, y)

  w2v_weights, vocab_size, embedding_size= BILSTM.word2vec(X_train)


  tokenizer= BILSTM.tokenizer_fit(X_train,vocab_size)
  
  train_padded=  BILSTM.tokenizer_transform(X_train, tokenizer, max_length,padding_type )
  val_padded=BILSTM.tokenizer_transform(X_val, tokenizer, max_length,padding_type )
  test_padded=BILSTM.tokenizer_transform(X_test, tokenizer, max_length,padding_type )

  label_tokenizer=  BILSTM.label_tokeniz_fit(y)

  train_labels_seq=BILSTM.label_tokeniz_transform(y_train,label_tokenizer )
  val_labels_seq = BILSTM.label_tokeniz_transform(y_val,label_tokenizer )
  test_labels_seq = BILSTM.label_tokeniz_transform(y_test,label_tokenizer )

  model = BILSTM.create_model(vocab_size, embedding_size, max_length, w2v_weights )
  BILSTM.plotModel(model)
  print('model ploted')
  history = model.fit(x= train_padded,y= train_labels_seq, 
                     validation_data=(val_padded,val_labels_seq), epochs=EPOCH)


  print("done")
  filePath= "../model/lstm"
  LogisticRegression.saveModel(filePath, model)
if(__name__ == "__main__"):
    fetech_clean_preprocess()
    ML() 
    DL()






