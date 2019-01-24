#Preprocessing on Data

import codecs
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

citation = ""
citation_with_context = ""
texts = []
texts_with_context = []
polarities = []
purposes = []

# import data
#Preprocessing: remove author names, years, email, url etc from original texts.
email_regex = r'[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z]{1,13}\.(?:com|cn|net)'
url_regex = r"\"?http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\"?"

for line in codecs.open("./Data/annotated_sentences.txt", "r", "utf-8", 'ignore').readlines():
    parts = line.split('\t')
    if parts[11].strip() != "0" and parts[12].strip() != "0":
        citation_with_context = parts[3] + " "+ parts[5] + " " + parts [7] + " " + parts[9]
        citation_with_context = re.sub(r'<[A-Z]+>.*?</[A-Z]+>', "",citation_with_context)
        citation_with_context = re.sub(url_regex, "", citation_with_context)
        texts_with_context.append(citation_with_context)
        parts[5] = re.sub(r'<[A-Z]+>.*?</[A-Z]+>', "",parts[5])
        parts[5] = re.sub(url_regex, "", parts[5])
        texts.append(parts[5])
        purposes.append(int(parts[11].strip()))
        polarities.append(int(parts[12].strip()))

preprocessed_text_with_context = []
# Preprocessing: Lemmatization
# Preprocessing: remove stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
print(stop_words)
for sen in texts_with_context:
    word_tokens = word_tokenize(sen)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = ""
    for w in word_tokens:
        w = lemmatizer.lemmatize(w)
        w = w.lower();
        if w not in stop_words and len(w)>1 and len(w)<40:
            if filtered_sentence == "":
                filtered_sentence = w
            else:
                filtered_sentence = filtered_sentence + " " + w
    print(filtered_sentence)
    preprocessed_text_with_context.append(filtered_sentence)

print(texts[0])
print(len(texts_with_context))
print(len(preprocessed_text_with_context))
print(len(polarities))
print(len(purposes))

#save
with open('./Pickle_Data/citation_with_context.pk', 'wb') as f:
    pickle.dump(texts_with_context, f)

with open('./Pickle_Data/citation.pk', 'wb') as f:
    pickle.dump(texts, f)

with open('./Pickle_Data/pre_citation_with_context.pk', 'wb') as f:
    pickle.dump(preprocessed_text_with_context, f)

with open('./Pickle_Data/polarities.pk', 'wb') as f:
    pickle.dump(polarities, f)

with open('./Pickle_Data/purposes.pk', 'wb') as f:
    pickle.dump(purposes, f)

# Check whether data are saved
# with open('C:/Users/songi/PycharmProjects/MasterThesis/purposes.pk', 'rb') as f:
#     data = pickle.load(f)
#     print(data)