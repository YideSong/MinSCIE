import sys
from gensim.models import Word2Vec, KeyedVectors
import nltk
import pickle
import numpy as np

# load the Stanford GloVe model
filename = 'C:/Users/songi/PycharmProjects/Model/acl_vectors_glove_300d.txt.word2vec'
print('loading model, model file: ', filename)
model = KeyedVectors.load_word2vec_format(filename, binary=False)

#Some example of word embeding
print('Examples:')
print(model.most_similar('cat'))

with open('C:/Users/songi/PycharmProjects/Master_Thesis/Pickle_Data/local_Model.pk', 'wb') as f:
    pickle.dump(model, f)
print("--------Vectors saved in local------------")