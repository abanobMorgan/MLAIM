# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy  as np
import scipy  as sc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('classic')
# %matplotlib inline
import pickle

import tensorflow as tf
tf.test.gpu_device_name() 
words2Vec_100F ="../model/dialect_dataset_words2Vec_100F.txt"
dataset_v_prep = "../data/mergedData.txt"
word2vec_twitter_Model="../model/word2vec_twitter_Model"

# reading Preprocced data
df_preprocced = pd.read_csv(dataset_v_prep)
df_preprocced.shape

df_preprocced = df_preprocced.dropna()
df_preprocced.sample(5)



X_data, y_data = np.array(df_preprocced['text']), np.array(df_preprocced['dialect'])

from gensim.models import word2vec
Embedding_dimensions = 100

# Creating Word2Vec training dataset.
toknized_data = list(map(lambda x: x.split(), X_data))
len(toknized_data)

# Defining the model and training it.
word2vec_model = word2vec.Word2Vec(toknized_data,
                 vector_size=Embedding_dimensions,
                 workers=8,
                 min_count=1)
word2vec_model.wv.save(word2vec_twitter_Model)

print("Vocabulary Length:", len(word2vec_model.wv.key_to_index))

def get_wordVec_Mean(tokens, size):
  count = 0
  vec = np.zeros(size).reshape((1,size))
  for word in tokens:
    try:
      vec += word2vec_model.wv[word].reshape((1,size))
      count += 1
    except KeyError:
      continue
  if count != 0:
    vec = vec/(count)
  return vec

# Creating word2vec Features
wordVecs_arrays = np.zeros((len(toknized_data),100))
for i in range(len(toknized_data)):
  wordVecs_arrays[i:] = get_wordVec_Mean(toknized_data[i], 100)
  if i% 10000 == len(toknized_data)%10000: 
    print(i)
wordvec_df = pd.DataFrame(wordVecs_arrays)
wordvec_df.shape

wordvec_df = pd.DataFrame(wordVecs_arrays)
wordvec_df

wordvec_df.to_csv(words2Vec_100F)
wordvec_df = pd.read_csv(words2Vec_100F, index_col=[0])
wordvec_df.shape

# Scaling Data and Normlization
from sklearn import preprocessing
# Scaling
scaled_Data = preprocessing.StandardScaler().fit_transform(wordvec_df)
scaled_Data = pd.DataFrame(scaled_Data)

#Normlization
normalized_Data = preprocessing.normalize(scaled_Data)
normalized_Data = pd.DataFrame(normalized_Data)

# Label Encoding for Classes
y_classes,_ = pd.factorize(y_data)
y_classes = pd.DataFrame(y_classes)

#Splitting Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(normalized_Data, y_classes, test_size = 0.25, shuffle=y_classes, random_state=100)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



# Defining the model input length.
input_length = 200
Embedding_dimensions = 100
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Label Encoding for Classes
from tensorflow.keras.utils import to_categorical
y_classes,_ = pd.factorize(y_data)
y_classes = pd.DataFrame(y_classes)
y = to_categorical(y_classes)

vocab_length = 247166

tokenizer = Tokenizer(filters="#", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(X_data)
tokenizer.num_words = vocab_length
print("Tokenizer vocab length:", vocab_length)

# Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X_data, y,test_size = 0.25, random_state = 0, stratify=y)

# Padding and Tokenizing Preprocced Data
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=input_length)
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test) , maxlen=input_length)

print("X_train.shape:", X_train.shape)
print("X_test.shape :", X_test.shape)

# Embedding Matrix to be Fed for Deep Learning Model
embedding_matrix = np.zeros((vocab_length, Embedding_dimensions))

for word, token in tokenizer.word_index.items():
    if word2vec_model.wv.__contains__(word):
        embedding_matrix[token] = word2vec_model.wv.__getitem__(word)

print("Embedding Matrix Shape:", embedding_matrix.shape)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding

# Building Deep Learning Model
def getModel():
    embedding_layer = Embedding(input_dim = vocab_length,
                                output_dim = Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(100, dropout=0.2, return_sequences=True)),
        Bidirectional(LSTM(100, dropout=0.2, return_sequences=True)),
        Conv1D(100, 5, activation='tanh'),
        GlobalMaxPool1D(),
        Dense(25, activation='tanh'),
        Dense(25, activation='tanh'),
        Dense(18, activation='softmax'),
    ],
    name="Sentiment_Model")
    return model

# Model Summary
training_model = getModel()
training_model.summary()

# Define CallBacks
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, cooldown=0),
             EarlyStopping(monitor='val_accuracy', min_delta=1e-7, patience=8)]

# Model Compile 
training_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Model Training

history = training_model.fit(X_train, y_train, batch_size=512, epochs=80, validation_split=0.1, callbacks=callbacks, verbose=1,)

history_Model="../model/history.pickle"
Tokenizer_Model="../model/Sentiment-BiLSTM"
Sentiment_Model="../model/word2vec_twitter_Model"

# Saving Nodel History
with open(history_Model, 'wb') as file:
  pickle.dump(history.history, file)

# Saving Models

# Saving the tokenizer
with open(Tokenizer_Model, 'wb') as file:
    pickle.dump(tokenizer, file)

# Saving the TF-Model.
training_model.save(Sentiment_Model)
training_model.save_weights(f"{Sentiment_Model}/weights")

# load history
history_file = open(history_Model, 'rb')
loaded_history = pickle.load(history_file)
