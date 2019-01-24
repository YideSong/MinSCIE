#Word Embedding: convert sentences to vectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
import gensim
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

# glove_input_file = 'glove.6B.300d.txt'
# word2vec_output_file = 'glove.6B.300d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)

# load the Stanford GloVe model
filename = '../Model/acl_vectors_glove_300d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

#Some example of word embeding
print(model.most_similar('algorithms'))
print(len(model['algorithms']))
print(model.most_similar('cat'))

# Calculate vector for a example sentence
example_sentence = "example sentence with word embeding"
print("Calculate vector for a example sentence: " + example_sentence)
token_sen = nltk.word_tokenize(example_sentence)
sen_len = 0
a = np.zeros(300)
for token in token_sen:
    if token in model.wv.vocab:
        sen_len = sen_len + 1
        a = a + model[token]
        print(model[token][0])
print(len(a))
a =a/sen_len
print(a[0])


#read preprocessed data from local
with open('./Pickle_Data/citation_with_context.pk', 'rb') as f:
    texts_with_context = pickle.load(f)
#    print(texts_with_context)
with open('./Pickle_Data/pre_citation_with_context.pk', 'rb') as f:
    pre_texts_with_context = pickle.load(f)
#    print(pre_texts_with_context)
with open('./Pickle_Data/citation.pk', 'rb') as f:
    texts = pickle.load(f)
#    print(texts)
with open('./Pickle_Data/polarities.pk', 'rb') as f:
    polarities = pickle.load(f)
#    print(polarities)
with open('./Pickle_Data/purposes.pk', 'rb') as f:
    purposes = pickle.load(f)
#    print(purposes)


citation_X = texts
citation_with_context_X = texts_with_context
pre_citation_with_context_X = pre_texts_with_context
polarity_Y = polarities
purpose_Y = purposes

citation_X = np.asarray(citation_X)
citation_with_context_X = np.asarray(citation_with_context_X)
pre_citation_with_context_X = np.asarray(pre_citation_with_context_X)
#print(citation_X)
print("------------Example of citations and its length:---------------")
print(len(citation_X))
print(citation_X[0])
print("------------Example of citations with contexts and its length:------------")
print(len(citation_with_context_X))
print(citation_with_context_X[0])
print("------------Example of citations with contexts and preprocessing and its length:------------")
print(len(pre_citation_with_context_X))
print(pre_citation_with_context_X[0])

print("---------create tfidf matrix-------------------")
vectorizer = CountVectorizer()
count = vectorizer.fit_transform(citation_with_context_X)
#print(vectorizer.get_feature_names())
#print(vectorizer.vocabulary_)
#print(count.toarray())
transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(count)
tfidf_matrix = tfidf_matrix.toarray()
print(len(tfidf_matrix))

# Calculate for each citation (whiout contexts) a Vector, save those vectors in a List -> vector_citations_X
# vector_citations_X = []
# sen_index = 0
# tfidf = 0
# vocabulary_index = 0
# for sen in citation_X:
#     token_sen = nltk.word_tokenize(sen)
#     sum_tfidt  = 0
#     vec_sen = np.zeros(300)
#     for token in token_sen:
#         if token in model.wv.vocab and token in vectorizer.get_feature_names():
#             vocabulary_index = vectorizer.vocabulary_[token]
#             tfidf = tfidf_matrix[sen_index][vocabulary_index]
#             sum_tfidt = sum_tfidt + tfidf
#             vec_sen = vec_sen + model[token] * tfidf #each vectors weighted by the tfidf value
#     #print(len(vec_sen))
#     vec_sen = vec_sen / sum_tfidt
#     vector_citations_X.append(vec_sen)
#     sen_index =sen_index + 1
# print("--------Number of citations converted to vectors: ------------")
# print(len(vector_citations_X))
# #print("-------Example vector for first citation: --------------------")
# #print(vector_citation_with_contexts_X[0])

# Calculate for each citations (with contexts) a Vector, save those vectors in a List -> vector_citation_with_contexts_X
vector_citation_with_contexts_X = []
sen_index = 0
vocabulary_index = 0
for sen in citation_with_context_X:
    token_sen = nltk.word_tokenize(sen)
    vec_sen = np.zeros(300)
    sen_len=0
    for token in token_sen:
        if token in model.wv.vocab:
            sen_len=sen_len+1
            vec_sen = vec_sen + model[token]
    #print(len(vec_sen))
    vec_sen = vec_sen / sen_len
    vector_citation_with_contexts_X.append(vec_sen)
    sen_index = sen_index + 1
print("--------Number of citations converted to vectors: ------------")
print(len(vector_citation_with_contexts_X))
#print("-------Example vector for first citation: --------------------")
#print(vector_citation_with_contexts_X[0])

# # Calculate for each citations (with contexts and preprocessing) a Vector, save those vectors in a List -> vector_citation_with_contexts_X
# vector_pre_citation_with_contexts_X = []
# sen_index = 0
# tfidf = 0
# vocabulary_index = 0
# for sen in pre_citation_with_context_X:
#     token_sen = nltk.word_tokenize(sen)
#     sum_tfidf = 0
#     vec_sen = np.zeros(300)
#     for token in token_sen:
#         if token in model.wv.vocab and token in vectorizer.get_feature_names():
#             vocabulary_index = vectorizer.vocabulary_[token]
#             tfidf = tfidf_matrix[sen_index][vocabulary_index]
#             sum_tfidf = sum_tfidf + tfidf
#             vec_sen = vec_sen + model[token] * tfidf
#     #print(len(vec_sen))
#     vec_sen = vec_sen / sum_tfidf
#     vector_pre_citation_with_contexts_X.append(vec_sen)
#     sen_index =sen_index + 1
# print("--------Number of citations converted to vectors: ------------")
# print(len(vector_pre_citation_with_contexts_X))
# #print("-------Example vector for first citation: --------------------")
# #print(vector_citation_with_contexts_X[0])

# # Save vectors of each citations to local.
# with open('./Pickle_Data/citation_vec.pk', 'wb') as f:
#     pickle.dump(vector_citations_X, f)
# print("--------Vectors saved in local------------")

# Save vectors of each citations (with contexts) to local.
with open('./Pickle_Data/citation_with_context_vec.pk', 'wb') as f:
    pickle.dump(vector_citation_with_contexts_X, f)
print("--------Vectors saved in local------------")

# # Save vectors of each citations (with contexts) to local.
# with open('./Pickle_Data/pre_citation_with_context_vec.pk', 'wb') as f:
#     pickle.dump(vector_pre_citation_with_contexts_X, f)
# # with open('C:/Users/songi/PycharmProjects/MasterThesis/citation_with_context_vec.pk', 'rb') as f:
# #     data = pickle.load(f)
# #     print(data[0])
# print("--------Vectors saved in local------------")