import codecs
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import itertools
from sklearn.utils import shuffle



#read preprocessed data from local
with open('./Pickle_Data/citation_with_context_vec.pk', 'rb') as f:
    vec_texts_with_context = pickle.load(f)
#    print(texts_with_context)
with open('./Pickle_Data/pre_citation_with_context_vec.pk', 'rb') as f:
    vec_pre_texts_with_context = pickle.load(f)
#
with open('./Pickle_Data/citation_vec.pk', 'rb') as f:
    vec_texts = pickle.load(f)
#    print(texts)
with open('./Pickle_Data/polarities.pk', 'rb') as f:
    polarities = pickle.load(f)
#    print(polarities)
with open('./Pickle_Data/purposes.pk', 'rb') as f:
    purposes = pickle.load(f)
#    print(purposes)

citation_X = vec_texts
citation_with_context_X = vec_texts_with_context
pre_citation_with_context_X = vec_pre_texts_with_context
polarity_Y = polarities
purpose_Y = purposes

#change to array
citation_X = np.asarray(citation_X)
citation_with_context_X = np.asarray(citation_with_context_X)
pre_citation_with_context_X = np.asarray(pre_citation_with_context_X)
polarity_Y = np.asarray(polarity_Y)
purpose_Y = np.asarray(purpose_Y)
citation_with_context_X.reshape(1,-1)
#print(citation_X)
#print("------------Example of citations and its length:---------------")
#print(len(citation_X))
#print(citation_X[0])
#print("------------Example of citations with contexts and its length:------------")
#print(len(citation_with_context_X))


#change NaN element to 0
nan_element = []
#remove nan in data
for i in range(len(citation_X)):
    sample=citation_X[i]
    for j in range(len(sample)):
        if np.isnan(sample[j]):
            sample[j]=0
            nan_element.append(i)
            #break
# print(nan_element)
# for i in nan_element:
#     citation_X = np.delete(citation_X,i,axis = 0)
#     polarity_Y = np.delete(polarity_Y,i,axis = 0)
#     purpose_Y = np.delete(purpose_Y,i,axis = 0)


for i in range(len(citation_with_context_X)):
    sample=citation_with_context_X[i]
    for j in range(len(sample)):
        if np.isnan(sample[j]):
            sample[j]=0

for i in range(len(pre_citation_with_context_X)):
    sample = pre_citation_with_context_X[i]
    for j in range(len(sample)):
        if np.isnan(sample[j]):
            sample[j] = 0


#shuffle the data
citation_with_context_X, polarity_Y, purpose_Y= shuffle(citation_with_context_X, polarity_Y,purpose_Y, random_state=0)


# Use cross validation to evaluate the model on all data (Train and test)
kf = KFold(n_splits=10, shuffle=False)
clf = svm.SVC(kernel='rbf', C=80, gamma=0.4)
accuracy_scores = []
precision_scores = []
recall_scores =[]
fscores = []
for k, (train, test) in enumerate(kf.split(citation_with_context_X, polarity_Y)):
    clf.fit(citation_with_context_X[train], polarity_Y[train])
    result = clf.predict(citation_with_context_X[test])
    accuracy_scores.append(accuracy_score(polarity_Y[test], result))
    precision_scores.append(precision_score(polarity_Y[test], result, average="macro"))
    recall_scores.append(recall_score(polarity_Y[test], result, average="macro"))
    fscores.append(f1_score(polarity_Y[test], result, average="macro"))
    print("[INFO] fold %s, score: %s " % (k, precision_recall_fscore_support(polarity_Y[test], result, average="macro" )))

print("Accuracy mean: %s, std. deviation: %s" %(np.mean(accuracy_scores)*100.0,np.std(accuracy_scores)*100.0))
print("precision_scores mean: %s, std. deviation: %s" %(np.mean(precision_scores)*100.0,np.std(precision_scores)*100.0))
print("recall_scores mean: %s, std. deviation: %s" %(np.mean(recall_scores)*100.0,np.std(recall_scores)*100.0))
print("fscores mean: %s, std. deviation: %s" %(np.mean(fscores)*100.0,np.std(fscores)*100.0))

f = open('Pickle_Data/svm_polarity.pk','wb')
clf.fit(citation_with_context_X, polarity_Y)
print(citation_with_context_X[1].reshape(1,-1))
result = clf.predict(citation_with_context_X[13].reshape(1,-1))
print(result)
pickle.dump(clf,f)
f.close()



kf = KFold(n_splits=10, shuffle=False)
clf = svm.SVC(kernel='rbf', C=75, gamma=1.1)
accuracy_scores = []
precision_scores = []
recall_scores =[]
fscores = []
for k, (train, test) in enumerate(kf.split(citation_with_context_X, purpose_Y)):
    clf.fit(citation_with_context_X[train], purpose_Y[train])
    result = clf.predict(citation_with_context_X[test])
    accuracy_scores.append(accuracy_score(purpose_Y[test], result))
    precision_scores.append(precision_score(purpose_Y[test], result, average="macro"))
    recall_scores.append(recall_score(purpose_Y[test], result, average="macro"))
    fscores.append(f1_score(purpose_Y[test], result, average="macro"))
    print("[INFO] fold %s, score: %s " % (k, precision_recall_fscore_support(purpose_Y[test], result, average="macro" )))

print("Accuracy mean: %s, std. deviation: %s" %(np.mean(accuracy_scores)*100.0,np.std(accuracy_scores)*100.0))
print("precision_scores mean: %s, std. deviation: %s" %(np.mean(precision_scores)*100.0,np.std(precision_scores)*100.0))
print("recall_scores mean: %s, std. deviation: %s" %(np.mean(recall_scores)*100.0,np.std(recall_scores)*100.0))
print("fscores mean: %s, std. deviation: %s" %(np.mean(fscores)*100.0,np.std(fscores)*100.0))


f = open('Pickle_Data/svm_purpose.pk','wb')
clf.fit(citation_with_context_X, purpose_Y)
result = clf.predict(citation_with_context_X[13].reshape(1,-1))
print(result)
pickle.dump(clf,f)
f.close()







