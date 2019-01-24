import time
import codecs
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def main():
    texts = []
    polarities = []

    # import data
    for line in codecs.open("./Data/annotated_sentences.txt", "r", "utf-8", 'ignore').readlines():
        parts = line.split('\t')
        if parts[12].strip() !="0":
            texts.append(parts[5])
            polarities.append(parts[12].strip())
    print("[INFO] Imported %s citation contexts and %s polarities." % (len(texts), len(polarities)))
    print("[INFO] Example context:\n %s" % (texts[0]))
    print("[INFO] Has a polarity value of %s" % (polarities[0]))
    print(set(polarities))

    # extract features
    count_vect = CountVectorizer(tokenizer=LemmaTokenizer())
    x_counts = count_vect.fit_transform(texts)
    print(x_counts)
    tfidf_transformer = TfidfTransformer()
    x_tfidf = tfidf_transformer.fit_transform(x_counts)

    # convert to numpy structures
    x = x_tfidf.toarray()
    y = np.asarray(polarities)

    # train classifier
    kf = KFold(n_splits=10, shuffle=True)
    clf = svm.LinearSVC()
    for k, (train, test) in enumerate(kf.split(x, y)):
        clf.fit(x[train], y[train])
        #print("[INFO] fold %s, score: %s " % (k, clf.score(x[test], y[test])))
        #print(train)
        #print(test)
        result = clf.predict(x[test])
        print("[INFO] fold %s, score: %s " % (k, precision_recall_fscore_support(y[test], result, average="macro" )))



if __name__ == "__main__":
    print("[INFO] Pipeline started")
    start_time = time.time()
    main()
    print("[INFO] Total processing time: %s seconds" % (time.time() - start_time))