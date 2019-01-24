import codecs
import nltk
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def main():
    texts_polarities = []
    texts_purposes = []
    texts = []
    polarities = []
    purposes = []
    polarities2 = []
    purposes2 = []
    vector_citation_x = []
    data_number = 0
    polarity_information = {"positive": 0, "neutral": 0, "negative": 0}
    purpose_information = {"Criticizing": 0, "Comparison": 0, "Use": 0, "Substantiating": 0, "Basis": 0, "Neutral": 0}

    # import data
    for line in codecs.open("./Data/annotated_sentences.txt", "r", "utf-8", 'ignore').readlines():
        data_number = data_number + 1
        parts = line.split('\t')
        if parts[12].strip() != "0":
            texts_polarities.append(parts[5])
            polarities.append(parts[12].strip())
        if parts[11].strip() != "0":
            texts_purposes.append(parts[5])
            purposes.append(parts[11].strip())
        if parts[11].strip() != "0" and parts[12].strip() != "0":
            texts.append(parts[5])
            purposes2.append(int(parts[11].strip()))
            polarities2.append(int(parts[12].strip()))

    citation_X = texts
    polarity_y = polarities2
    purpose_y = purposes2

    citation_X = np.asarray(citation_X)
    print(citation_X)

    #tok_corp=[nltk.word_tokenize(sent) for sent in citation_X]
    #model = gensim.models.Word2Vec(tok_corp, min_count=1, size=32)

    #model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/songi/PycharmProjects/MasterThesis/GoogleNews-vectors-negative300.bin', binary=True)
    #model.save('word2vec.model')
    model = KeyedVectors.load('word2vec.model')
    print(model.most_similar('algorithms'))
    print(len(model['algorithms']))
    print(model.most_similar('cat'))


    for sen in citation_X:
        token_sen = nltk.word_tokenize(sen)
        sen_len = 0
        a = np.zeros(300)
        for token in token_sen:
            if token in model.wv.vocab:
                sen_len = sen_len + 1
                a = a + model[token]
        print(len(a))
        a = a / sen_len
        print(a)
        vector_citation_x.append(a)

    print(vector_citation_x)







if __name__ == "__main__":
    print("[INFO] Pipeline started")
    main()