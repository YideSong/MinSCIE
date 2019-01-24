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

#Create svm model for label: polarity
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(citation_with_context_X, polarity_Y, test_size=0.1, random_state=10)

# Define different values for the hyperparameter optimization
kernel_list = ['rbf', 'linear']
c_list = [ 80, 85, 90, 95, 100]
g_list = [ 0.3, 0.4, 0.5, 0.75, 1, 10]
# Build cartesian product
configs = list(itertools.product(c_list, g_list,kernel_list))

current_best_f1 = 0.0
best_kernel = ""
best_c = None
best_g = None

for c, g, kernel in configs:
    kf = KFold(n_splits=10, shuffle=False)
    clf = svm.SVC(kernel=kernel, C=c, gamma=g)
    fscores = []
    current_f1 = 0
    for k, (train, dev) in enumerate(kf.split(X_train, y_train)):
        clf.fit(X_train[train], y_train[train])
        result = clf.predict(X_train[dev])
        fscores.append(f1_score(y_train[dev], result, average="macro"))
        #print("[INFO] fold %s, score: %s " % (k, precision_recall_fscore_support(y_train[dev], result, average="macro" )))
    print("[INFO] current kernel: %s, current c: %s, current g: %s " % (kernel, c, g))
    current_f1 = np.mean(fscores)
    print("[INFO] Current average fscore: %s" %(current_f1))
    if(current_f1 > current_best_f1):
        current_best_f1 = current_f1
        best_c = c
        best_g = g
        best_kernel = kernel
print(best_kernel)
print(best_g)
print(best_c)
print(current_best_f1)

#Test on test data set
clf = svm.SVC(kernel=best_kernel, C=best_c, gamma=best_g)
clf.fit(X_train, y_train)
result = clf.predict(X_test)
print(precision_recall_fscore_support(y_test, result, average="macro" ))

# Use cross validation to evaluate the model on all data (Train and test)
kf = KFold(n_splits=10, shuffle=False)
clf = svm.SVC(kernel=best_kernel, C=best_c, gamma=best_g)
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


#Create svm model for label: purpose
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(citation_with_context_X, purpose_Y, test_size=0.1, random_state=10)


# Define different values for the hyperparameter optimization
kernel_list = ['rbf', 'linear']
c_list = [ 80, 85, 90, 95,100]
g_list = [ 0.3, 0.4, 0.5, 0.75, 1, 10]
# Build cartesian product
configs = list(itertools.product(c_list, g_list,kernel_list))

current_best_f1 = 0.0
best_kernel = ""
best_c = None
best_g = None

for c, g, kernel in configs:
    kf = KFold(n_splits=10, shuffle=False)
    clf = svm.SVC(kernel=kernel, C=c, gamma=g)
    fscores = []
    current_f1 = 0
    for k, (train, dev) in enumerate(kf.split(X_train, y_train)):
        clf.fit(X_train[train], y_train[train])
        result = clf.predict(X_train[dev])
        fscores.append(f1_score(y_train[dev], result, average="macro"))
        #print("[INFO] fold %s, score: %s " % (k, precision_recall_fscore_support(y_train[dev], result, average="macro" )))
    print("[INFO] current kernel: %s, current c: %s, current g: %s " % (kernel, c, g))
    current_f1 = np.mean(fscores)
    print("[INFO] Current average fscore: %s" %(current_f1))
    if(current_f1 > current_best_f1):
        current_best_f1 = current_f1
        best_c = c
        best_g = g
        best_kernel = kernel
print(best_kernel)
print(best_g)
print(best_c)
print(current_best_f1)

#Test on test data set
clf = svm.SVC(kernel=best_kernel, C=best_c, gamma=best_g)
clf.fit(X_train, y_train)
result = clf.predict(X_test)
print(precision_recall_fscore_support(y_test, result, average="macro" ))

# Use cross validation to evaluate the model on all data (Train and test)
kf = KFold(n_splits=10, shuffle=False)
clf = svm.SVC(kernel=best_kernel, C=best_c, gamma=best_g)
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


#test
#Backup: old models
# # train classifier svm
# print("------SVM model: svm.LinearSVC() with Kfold. input: vector of each citation, label: polarities-------")
# kf = KFold(n_splits=10, shuffle=False)
# #clf = OneVsRestClassifier(svm.SVC(kernel='linear'))
# #clf = svm.LinearSVC()
# clf = svm.SVC(C=100.0, decision_function_shape='ovo', kernel='linear')
# accuracy_scores = []
# precision_scores = []
# recall_scores =[]
# fscores = []
# for k, (train, test) in enumerate(kf.split(citation_with_context_X, polarity_Y)):
#     clf.fit(citation_with_context_X[train], polarity_Y[train])
#     result = clf.predict(citation_with_context_X[test])
#     accuracy_scores.append(accuracy_score(polarity_Y[test], result))
#     precision_scores.append(precision_score(polarity_Y[test], result, average="macro"))
#     recall_scores.append(recall_score(polarity_Y[test], result, average="macro"))
#     fscores.append(f1_score(polarity_Y[test], result, average="macro"))
#     print("[INFO] fold %s, score: %s " % (k, precision_recall_fscore_support(polarity_Y[test], result, average="macro" )))
#
# print("Accuracy mean: %s, std. deviation: %s" %(np.mean(accuracy_scores)*100.0,np.std(accuracy_scores)*100.0))
# print("precision_scores mean: %s, std. deviation: %s" %(np.mean(precision_scores)*100.0,np.std(precision_scores)*100.0))
# print("recall_scores mean: %s, std. deviation: %s" %(np.mean(recall_scores)*100.0,np.std(recall_scores)*100.0))
# print("fscores mean: %s, std. deviation: %s" %(np.mean(fscores)*100.0,np.std(fscores)*100.0))


# # train classifier svm
# print("------SVM model: svm.LinearSVC() with Kfold. input: vector of each citation, label: purpose-------")
# kf = KFold(n_splits=10, shuffle=False)
# #clf = OneVsRestClassifier(svm.SVC(kernel='linear'))
# #clf = svm.LinearSVC()
# clf = svm.SVC(C=100.0, decision_function_shape='ovo', kernel='linear')
# accuracy_scores = []
# precision_scores = []
# recall_scores =[]
# fscores = []
# for k, (train, test) in enumerate(kf.split(citation_with_context_X, purpose_Y)):
#     clf.fit(citation_with_context_X[train], purpose_Y[train])
#     result = clf.predict(citation_with_context_X[test])
#     accuracy_scores.append(accuracy_score(purpose_Y[test], result))
#     precision_scores.append(precision_score(purpose_Y[test], result, average="macro"))
#     recall_scores.append(recall_score(purpose_Y[test], result, average="macro"))
#     fscores.append(f1_score(purpose_Y[test], result, average="macro"))
#     print("[INFO] fold %s, score: %s " % (k, precision_recall_fscore_support(purpose_Y[test], result, average="macro" )))

# print("Accuracy mean: %s, std. deviation: %s" %(np.mean(accuracy_scores)*100.0,np.std(accuracy_scores)*100.0))
# print("precision_scores mean: %s, std. deviation: %s" %(np.mean(precision_scores)*100.0,np.std(precision_scores)*100.0))
# print("recall_scores mean: %s, std. deviation: %s" %(np.mean(recall_scores)*100.0,np.std(recall_scores)*100.0))
# print("fscores mean: %s, std. deviation: %s" %(np.mean(fscores)*100.0,np.std(fscores)*100.0))
#
# clf = svm.SVC(kernel='linear', C=100)
# scores = cross_val_score(clf, citation_with_context_X, polarity_Y, cv=10, scoring="recall_macro")
# print(scores)
# print(scores.mean())
# print(scores.std())


# grid_param = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
#               {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# gd_sr = GridSearchCV(estimator=svm.SVC(),  param_grid=grid_param)
# gd_sr.fit(citation_with_context_X, polarity_Y)
# print('Best score for data1:', gd_sr.best_score_)
# print('Best C:',gd_sr.best_estimator_.C)
# print('Best Kernel:',gd_sr.best_estimator_.kernel)
# print('Best Gamma:',gd_sr.best_estimator_.gamma)

# ParaC = np.array([1,0.1,0.01,0.001,0.0001,10, 100, 1000])
# # create and fit a ridge regression model, testing each alpha
# model = svm.LinearSVC()
# grid = GridSearchCV(estimator=model, param_grid=dict(C=ParaC))
# grid.fit(citation_with_context_X[train], polarity_Y[train])
# print(grid)
# # summarize the results of the grid search
# print(grid.best_score_)
# print(grid.best_estimator_.C)



# x_train1, x_test1, y_train1, y_test1 = train_test_split(citation_X, polarity_Y, random_state=0, train_size=0.9)
# x_train2, x_test2, y_train2, y_test2 = train_test_split(citation_with_context_X, polarity_Y, random_state=0, train_size=0.9)
# x_train3, x_test3, y_train3, y_test3 = train_test_split(pre_citation_with_context_X, polarity_Y, random_state=0, train_size=0.9)
# print("------SVM model: svm.LinearSVC(). input: vector of each citation, label: polarities-------")
# clf.fit(x_train1,y_train1)
# result = clf.predict(x_test1)
# print(precision_recall_fscore_support(y_test1, result, average="macro"))
# #
# # one vs the rest
# print("------SVM model: OneVsRestClassifier. input: vector of each citation, label: polarities-------")
# result = OneVsRestClassifier(svm.SVC(kernel='linear')).fit(x_train1,y_train1).predict(x_test1)
# print(precision_recall_fscore_support(y_test1, result, average="macro"))
# #print(result2)
#
# print("------SVM model: svm.LinearSVC(). input: vector of each citation with its contexts, label: polarities-------")
# clf.fit(x_train2,y_train2)
# result2 = clf.predict(x_test2)
# print(precision_recall_fscore_support(y_test2, result2, average="macro"))
#
# # one vs the rest
# print("------SVM model: OneVsRestClassifier. input: vector of each citation with its contexts, label: polarities-------")
# result2 = OneVsRestClassifier(svm.SVC(kernel='linear')).fit(x_train2,y_train2).predict(x_test2)
# print(precision_recall_fscore_support(y_test2, result2, average="macro"))
# #print(result2)
#
# print("------SVM model: svm.LinearSVC(). input: vector of each citation with its contexts and with preprocessing, label: polarities-------")
# clf.fit(x_train3,y_train3)
# result3 = clf.predict(x_test3)
# print(precision_recall_fscore_support(y_test3, result3, average="macro"))
#
# # one vs the rest
# print("------SVM model: OneVsRestClassifier. input: vector of each citation with its contexts, label: polarities-------")
# result3 = OneVsRestClassifier(svm.SVC(kernel='linear')).fit(x_train3,y_train3).predict(x_test3)
# print(precision_recall_fscore_support(y_test3, result3, average="macro"))




# #label binarizer
# lb = preprocessing.LabelBinarizer()
# lb.fit([1,2,3])
# lb.classes_
# polarity_Y = lb.fit_transform(polarity_Y)
# #print(polarity_Y)
#
# # one vs the rest
# print("------SVM model: OneVsRestClassifier. input: vector of each citations, binarizer label: polarities-------")
# x_train1, x_test1, y_train1, y_test1 = train_test_split(citation_X, polarity_Y, random_state=0, train_size=0.8)
# model = OneVsRestClassifier(svm.SVC(kernel='linear'))
# model.fit(x_train1,y_train1)
# result = model.predict(x_test1)
# print(precision_recall_fscore_support(y_test1, result, average="macro"))


