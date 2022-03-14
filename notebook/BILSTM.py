import pandas as pd
import numpy as np 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential

from keras.layers.core import Dropout
from keras.layers.core import Dense

from keras.layers import LeakyReLU
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
import gensim 
import keras 

import LogisticRegression



def word2vec(x_train): 


  w2v_model = gensim.models.Word2Vec(x_train, size=160, window=5, min_count=4, workers=8)

  # Retrieve the weights from the model. This is used for initializing the weights
  # in a Keras Embedding layer later
  w2v_weights = w2v_model.wv.vectors
  vocab_size, embedding_size = w2v_weights.shape

  print("Vocabulary Size: {} - Embedding Dim: {}".format(vocab_size, embedding_size))
  return w2v_weights, vocab_size, embedding_size


def tokenizer_fit(X_trian,vocab_size , OOV_tok='<OoV>' ): 
  tokenizer = Tokenizer(num_words=vocab_size, oov_token= OOV_tok)
  tokenizer.fit_on_texts(X_trian)
  return tokenizer
def tokenizer_transform(X, tokenizer, max_length,padding_type ): 

  word_index = tokenizer.word_index

  X_sequences = tokenizer.texts_to_sequences(X)
  X_padded = pad_sequences(X_sequences, padding= padding_type, maxlen= max_length)
  return X_padded

def label_tokeniz_fit(y): 

  label_tokenizer = Tokenizer()
  label_tokenizer.fit_on_texts(y)
  return label_tokenizer


def label_tokeniz_transform(y,label_tokenizer ): 

  train_labels_seq=np.array(label_tokenizer.texts_to_sequences(y))

  return train_labels_seq




def create_model(vocab_size, embedding_size, max_length, w2v_weights ): 
 
 
  model = Sequential([         # embedding layer
        Embedding(vocab_size, embedding_size, input_length= max_length, weights=[w2v_weights], trainable=True),

        Bidirectional(LSTM(150, recurrent_dropout=0.3,return_sequences=True)),
        Bidirectional(LSTM(70, recurrent_dropout=0.3)),

      # Classification head
      Dense(180, activation=LeakyReLU()),Dropout(.5),
      Dense(64, activation='relu'),Dropout(.5),
      Dense(19, activation='softmax')    
    ]) 

  model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) 

  model.summary()
  return model


def plotModel(model):
  """
  input: 
    model: 
  
  this function takes a model and plot it
  """
  keras.utils.vis_utils.plot_model(model, show_shapes=True)

