import sys
from gensim.models import Word2Vec, KeyedVectors
import nltk
import pickle
import numpy as np

# load the Stanford GloVe model
#filename = 'C:/Users/songi/PycharmProjects/Model/glove.6B.300d.txt.word2vec'
#print('loading model, model file: ', filename)
#model = KeyedVectors.load_word2vec_format(filename, binary=False)

with open('C:/Users/songi/PycharmProjects/Master_Thesis/Pickle_Data/local_Model.pk', 'rb') as f:
    model = pickle.load(f)



# Calculate for user input text a Vector,
user_input_text = sys.argv[1]
vector_user_input_text = np.zeros(300)
token_sen = nltk.word_tokenize(user_input_text)
print(token_sen)

sen_len = 0
for token in token_sen:
    if token in model.wv.vocab:
        sen_len = sen_len + 1
        vector_user_input_text = vector_user_input_text + model[token]
        #print(model[token][0])

vector_user_input_text = vector_user_input_text/sen_len
#print("vector: ", vector_user_input_text)


polarity_information = {"positive": 0, "neutral": 0, "negative": 0}
purpose_information = {"Criticizing": 0, "Comparison": 0, "Use": 0, "Substantiating": 0, "Basis": 0, "Neutral": 0}

f = open('C:/Users/songi/PycharmProjects/Master_Thesis/Pickle_Data/svm_polarity.pk','rb')
svm_model = pickle.load(f)
f.close()

result = svm_model.predict(vector_user_input_text.reshape(1,-1))
polarity = ""
if result == 1:
    polarity = "Neutral"
if result == 2:
    polarity = "Positive"
if result == 3:
    polarity == "Negative"
print(polarity)

f = open('C:/Users/songi/PycharmProjects/Master_Thesis/Pickle_Data/svm_purpose.pk','rb')
svm_model = pickle.load(f)
f.close()

result2 = svm_model.predict(vector_user_input_text.reshape(1,-1))
purpose = ""
if result == 1:
    purpose = "Criticizing"
if result == 2:
    purpose = "Comparison"
if result == 3:
    purpose == "Use"
if result == 4:
    purpose = "Substantiating"
if result == 5:
    purpose = "Basis"
if result == 6:
    purpose == "Neutral"
print(purpose)

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv[1]))