import pandas as pd
import numpy as np 
import csv 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from keras.layers.core import Dropout
from keras.layers.core import Dense

from keras.layers import LeakyReLU
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
import gensim 
import keras 

import LogisticRegression


df = pd.read_csv("/content/drive/MyDrive/mergedData.txt")
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



w2v_model = gensim.models.Word2Vec(X_train, size=160, window=5, min_count=4, workers=8)

# Retrieve the weights from the model. This is used for initializing the weights
# in a Keras Embedding layer later
w2v_weights = w2v_model.wv.vectors
vocab_size, embedding_size = w2v_weights.shape

print("Vocabulary Size: {} - Embedding Dim: {}".format(vocab_size, embedding_size))

tokenizer = Tokenizer(num_words=vocab_size, oov_token= OOV_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

trian_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(trian_sequences, padding= padding_type, maxlen= max_length)

val_sequences = tokenizer.texts_to_sequences(X_val)
val_padded = pad_sequences(val_sequences, padding= padding_type, maxlen= max_length)

test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, padding= padding_type, maxlen= max_length)

len(trian_sequences)

del X_test, X_val, X_train



label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(y_train)

train_labels_seq=np.array(label_tokenizer.texts_to_sequences(y_train))
val_labels_seq = np.array(label_tokenizer.texts_to_sequences(y_val))
test_labels_seq = np.array(label_tokenizer.texts_to_sequences(y_test))

y_train.shape, y_test.shape, vocab_size

def create_model(vocab_size, embedding_size, max_length, w2v_weights ): 
 
 
  model = Sequential([         # embedding layer
        Embedding(vocab_size, embedding_size, input_length= max_length, weights=[w2v_weights], trainable=True),

        Bidirectional(LSTM(150, recurrent_dropout=0.3,return_sequences=True)),
        Bidirectional(LSTM(70, recurrent_dropout=0.3)),

      # Classification head
      Dense(180, activation='relu'),Dropout(.5),
      Dense(64, activation='relu'),Dropout(.5),
      Dense(19, activation='softmax')    
    ]) 

  model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) 

  model.summary()
  return model

# 
def plotModel(model):
  """
  input: 
    model: 
  
  this function takes a model and plot it
  """
  keras.utils.vis_utils.plot_model(model, show_shapes=True)

def main(): 
  model = create_model(vocab_size, embedding_size, max_length, w2v_weights )
  plotModel(model)
  
  history = model.fit(x= train_padded,y= train_labels_seq, 
                     validation_data=(val_padded,val_labels_seq), epochs=EPOCH)



  filePath= "../model/lstm"
  LogisticRegression.saveModel(filePath, model)

"""
import pickle
filename= "lstm"
pickle.dump(model, open(filename, 'wb'))



test_z = vectr.transform([" احنا بيقنا الصبح استاذ مجدي يومك بيضحك"])

model.predict(test_z)

import pickle
filename= "ANN"
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(val_X,y_val)
print(result)

with open('vectorizer.pk', 'wb') as fin:
  pickle.dump(vectr, fin)
 
#exit()



from sklearn.decomposition import TruncatedSVD
features= 3240
pca = TruncatedSVD(features)
pca.fit(train_X)
train_X.shape
train_X= pca.transform(train_X)
val_X= pca.transform(val_X)
test_X= pca.transform(test_X)

train_X.shape, val_X.shape, test_X.shape"""