from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#read preprocessed data from local
with open('./Pickle_Data/citation_with_context.pk', 'rb') as f:
    texts_with_context = pickle.load(f)
    print(texts_with_context[1])
with open('./Pickle_Data/pre_citation_with_context.pk', 'rb') as f:
    pre_texts_with_context = pickle.load(f)
    print(pre_texts_with_context[1])
with open('./Pickle_Data/citation.pk', 'rb') as f:
    texts = pickle.load(f)
    print(texts[1])
with open('./Pickle_Data/polarities.pk', 'rb') as f:
    polarities = pickle.load(f)
#    print(polarities)
with open('./Pickle_Data/purposes.pk', 'rb') as f:
    purposes = pickle.load(f)
#    print(purposes)

citation_X = texts
citation_with_context_X = texts_with_context
polarity_Y = polarities
purpose_Y = purposes

citation_X = np.asarray(citation_X)
citation_with_context_X = np.asarray(citation_with_context_X)
polarity_Y = np.asarray(polarity_Y)
purpose_Y = np.asarray(purpose_Y)

example_document = ["I have an pen dog cat box.",
                    "I have an apple."]

print("---------create tfidf matrix for citations without contexts-------------------")
vectorizer = CountVectorizer()
count = vectorizer.fit_transform(citation_with_context_X)
#print(vectorizer.get_feature_names())
# if "yide" in vectorizer.get_feature_names():
#     print(vectorizer.vocabulary_["yide"])
print(count.toarray())
transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(count)
#print(tfidf_matrix.toarray())
#print(tfidf_matrix.toarray())


polarity_Y = np.asarray(polarities)
purpose_Y = np.asarray(purposes)
tfidf_matrix = tfidf_matrix.toarray()

# train classifier normal svm
kf = KFold(n_splits=10, shuffle=False)
clf = svm.LinearSVC()
accuracy_scores = []
precision_scores = []
recall_scores =[]
fscores = []

for k, (train, test) in enumerate(kf.split(tfidf_matrix, polarity_Y)):
    clf.fit(tfidf_matrix[train], polarity_Y[train])
    result = clf.predict(tfidf_matrix[test])
    accuracy_scores.append(accuracy_score(polarity_Y[test], result))
    precision_scores.append(precision_score(polarity_Y[test], result, average="macro"))
    recall_scores.append(recall_score(polarity_Y[test], result, average="macro"))
    fscores.append(f1_score(polarity_Y[test], result, average="macro"))
    print("[INFO] fold %s, score: %s " % (k, precision_recall_fscore_support(polarity_Y[test], result, average="macro" )))

print("Accuracy mean: %s, std. deviation: %s" % (np.mean(accuracy_scores) * 100.0, np.std(accuracy_scores) * 100.0))
print("precision_scores mean: %s, std. deviation: %s" % (np.mean(precision_scores) * 100.0, np.std(precision_scores) * 100.0))
print("recall_scores mean: %s, std. deviation: %s" % (np.mean(recall_scores) * 100.0, np.std(recall_scores) * 100.0))
print("fscores mean: %s, std. deviation: %s" % (np.mean(fscores) * 100.0, np.std(fscores) * 100.0))


# train classifier normal svm
kf = KFold(n_splits=10, shuffle=False)
clf = svm.LinearSVC()
accuracy_scores = []
precision_scores = []
recall_scores =[]
fscores = []

for k, (train, test) in enumerate(kf.split(tfidf_matrix, purpose_Y)):
    clf.fit(tfidf_matrix[train], purpose_Y[train])
    result = clf.predict(tfidf_matrix[test])
    accuracy_scores.append(accuracy_score(purpose_Y[test], result))
    precision_scores.append(precision_score(purpose_Y[test], result, average="macro"))
    recall_scores.append(recall_score(purpose_Y[test], result, average="macro"))
    fscores.append(f1_score(purpose_Y[test], result, average="macro"))
    print("[INFO] fold %s, score: %s " % (k, precision_recall_fscore_support(purpose_Y[test], result, average="macro" )))

print("Accuracy mean: %s, std. deviation: %s" % (np.mean(accuracy_scores) * 100.0, np.std(accuracy_scores) * 100.0))
print("precision_scores mean: %s, std. deviation: %s" % (np.mean(precision_scores) * 100.0, np.std(precision_scores) * 100.0))
print("recall_scores mean: %s, std. deviation: %s" % (np.mean(recall_scores) * 100.0, np.std(recall_scores) * 100.0))
print("fscores mean: %s, std. deviation: %s" % (np.mean(fscores) * 100.0, np.std(fscores) * 100.0))
#
#
# # x_train1, x_test1, y_train1, y_test1 = train_test_split(tfidf_matrix, polarity_Y, random_state=0, train_size=0.8)
# # print("------SVM model: svm.LinearSVC(). input: vector of each citation, label: polarities-------")
# # clf.fit(x_train1,y_train1)
# # result = clf.predict(x_test1)
# # print(precision_recall_fscore_support(y_test1, result, average="macro"))
# #
# # # one vs the rest
# # print("------SVM model: OneVsRestClassifier. input: vector of each citation, label: polarities-------")
# # result = OneVsRestClassifier(svm.SVC(kernel='linear')).fit(x_train1,y_train1).predict(x_test1)
# # print(precision_recall_fscore_support(y_test1, result, average="macro"))
# # #print(result2)
#
#
# # print("---------create tfidf matrix for citations with contexts-------------------")
# # vectorizer = CountVectorizer()
# # count = vectorizer.fit_transform(texts_with_context)
# # transformer = TfidfTransformer()
# # tfidf_matrix2 = transformer.fit_transform(count)
# # print(len(tfidf_matrix2.toarray()))
# # print(len(tfidf_matrix2.toarray()[2]))
# # tfidf_matrix2 = tfidf_matrix2.toarray()
# #
# # # train classifier svm
# # clf = svm.LinearSVC()
# # x_train2, x_test2, y_train2, y_test2 = train_test_split(tfidf_matrix2, polarity_Y, random_state=0, train_size=0.8)
# # print("------SVM model: svm.LinearSVC(). input: vector of each citation with context, label: polarities-------")
# # clf.fit(x_train2,y_train2)
# # result = clf.predict(x_test2)
# # print(precision_recall_fscore_support(y_test2, result, average="macro"))
# #
# # # one vs the rest
# # print("------SVM model: OneVsRestClassifier. input: vector of each citation with context, label: polarities-------")
# # result = OneVsRestClassifier(svm.SVC(kernel='linear')).fit(x_train2,y_train2).predict(x_test2)
# # print(precision_recall_fscore_support(y_test2, result, average="macro"))
#
#
#
# # print("---------create tfidf matrix for citations with contexts and preprocessing-------------------")
# # vectorizer = CountVectorizer()
# # count = vectorizer.fit_transform(pre_texts_with_context)
# # transformer = TfidfTransformer()
# # tfidf_matrix3 = transformer.fit_transform(count)
# # print(len(tfidf_matrix3.toarray()))
# # print(len(tfidf_matrix3.toarray()[2]))
# # tfidf_matrix3 = tfidf_matrix3.toarray()
# #
# # # train classifier svm
# # clf = svm.LinearSVC()
# # x_train3, x_test3, y_train3, y_test3 = train_test_split(tfidf_matrix3, polarity_Y, random_state=0, train_size=0.8)
# # print("------SVM model: svm.LinearSVC(). input: vector of each citation with context, label: polarities-------")
# # clf.fit(x_train3,y_train3)
# # result = clf.predict(x_test3)
# # print(precision_recall_fscore_support(y_test3, result, average="macro"))
# #
# # # one vs the rest
# # print("------SVM model: OneVsRestClassifier. input: vector of each citation with context, label: polarities-------")
# # result = OneVsRestClassifier(svm.SVC(kernel='linear')).fit(x_train3,y_train3).predict(x_test3)
# # print(precision_recall_fscore_support(y_test3, result, average="macro"))