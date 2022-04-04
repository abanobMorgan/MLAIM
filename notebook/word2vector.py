import pandas as pd
from gensim.models import word2vec
import numpy as np 

WORDSMODEL ="../model/word2vec_twitter_Model.model"
WORDS2VEC_100F = "../model/dialect_dataset_words2Vec_100F.txt"
DATAPATH = "../data/mergedData.txt"

def createW2Vmodel(): 
    df= pd.read_csv(DATAPATH)
    df = df.dropna()
    X_data, y_data = np.array(df['text']), np.array(df['dialect'])

    Embedding_dimensions = 100

    # Creating Word2Vec training dataset.
    toknized_data = list(map(lambda x: x.split(), X_data))
    try:
      word2vec_model = word2vec.Word2Vec.load(WORDSMODEL)
    except:
      # Defining the model and training it.
      word2vec_model = word2vec.Word2Vec(toknized_data,
                      vector_size=Embedding_dimensions,
                      workers=8,
                      min_count=1)

    print("Vocabulary Length:", len(word2vec_model.wv.key_to_index))

    word2vec_model.save(WORDSMODEL)
    print("model saved ")
    return toknized_data

# Prepare word2vec data set Features
def get_wordVec_Mean(word2vec_model, tokens, size):
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

def word2vec_Features(toknized_data, word2vec_model, maxlenght=100):
    try:
      df = pd.read_csv(WORDS2VEC_100F)
      print("model founded")
      return df
    except:
      # Creating word2vec Features
      wordVecs_arrays = np.zeros((len(toknized_data),100))

      for i in range(len(toknized_data)):
        wordVecs_arrays[i:] = get_wordVec_Mean(word2vec_model, toknized_data[i], maxlenght)
        if i% 10000 == len(toknized_data)%10000: 
          print(i)

      df = pd.DataFrame(wordVecs_arrays)
      df.to_csv(WORDS2VEC_100F)
      print("model saved ")
      return df

def embedding (tokenizer, word2vec_model): 
  embedding_dictionary = dict()
  with open(WORDS2VEC_100F,'r') as file:
      for line in file:
          values=line.split()
          word=values[0]
          vectors=np.asarray(values[1:],'float32')
          embedding_dictionary[word]=vectors
  file.close()

  MAXLEN=100
  vocab_length = len(tokenizer.word_index) + 1 
  Embedding_dimensions = 100

  
  embedding_matrix = np.zeros((vocab_length, Embedding_dimensions))
  for word, token in tokenizer.word_index.items():
      if word2vec_model.wv.__contains__(word):
          embedding_matrix[token] = word2vec_model.wv.__getitem__(word)
  print("Embedding Matrix Shape:", embedding_matrix.shape)
  return embedding_matrix

def get_embedding(tokenizer): 
  try:
    word2vec_model = word2vec.Word2Vec.load(WORDSMODEL)
    return embedding (tokenizer, word2vec_model)
  except:
    toknized_data= createW2Vmodel()
    word2vec_model = word2vec.Word2Vec.load(WORDSMODEL)
    print('model loaded into system')
    data = word2vec_Features(toknized_data, word2vec_model)
    return embedding (tokenizer, word2vec_model)
  
