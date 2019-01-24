# Data analysis, Baseline, evaluation on baseline

import time
import codecs
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


def main():
    texts_polarities = []
    texts_purposes = []
    texts = []
    polarities = []
    purposes = []
    polarities2 = []
    purposes2 = []
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
            if parts[12].strip() == "1":
                polarity_information["neutral"] += 1
            if parts[12].strip() == "2":
                polarity_information["positive"] += 1
            if parts[12].strip() == "3":
                polarity_information["negative"] += 1
            if parts[11].strip() == "1":
                purpose_information["Criticizing"] += 1
            if parts[11].strip() == "2":
                purpose_information["Comparison"] += 1
            if parts[11].strip() == "3":
                purpose_information["Use"] += 1
            if parts[11].strip() == "4":
                purpose_information["Substantiating"] += 1
            if parts[11].strip() == "5":
                purpose_information["Basis"] += 1
            if parts[11].strip() == "6":
                purpose_information["Neutral"] += 1
    print("-------------------------------statistic on data----------------------------------------")
    print("[INFO] Total data Number: %s" % data_number)
    print("[INFO] Data contains %s citations and %s polarities." % (len(texts_polarities), len(polarities)))
    print("[INFO] Data contains %s citations and %s purposes." % (len(texts_purposes), len(purposes)))
    print("[INFO] Data contains %s citation contexts and %s polarities and %s Purposes." % (len(texts), len(polarities2), len(purposes2)))
    print("[INFO] statistic on polarity %s" % polarity_information)
    print("[INFO] statistic on purpose %s" % purpose_information)
    print("-------------------------------Example----------------------------------------")
    print("[INFO] Example context:\n %s" % (texts[0]))
    print("[INFO] Has a polarity value of %s" % (polarities2[0]))

    citation_X = texts
    polarity_y = polarities2
    purpose_y = purposes2


    print("-------------------------------Baseline Majority-----------------------------")
    y1_result = []
    for i in polarity_y:
        y1_result.append(1)
    print("y1_result %s" % y1_result)

    y2_result = []
    for i in purpose_y:
        y2_result.append(6)
    print("y2_result %s" % y2_result)

    #print("purpose_y %s" %purpose_y)
    polarity_y = np.asarray(polarity_y)
    purpose_y = np.asarray(purpose_y)
    y1_result = np.asarray(y1_result)
    y2_result = np.asarray(y2_result)

    print("-------------------------------Evaluation on Majority-----------------------------")
    print("[INFO] Accuracy score for polarity: %s " % accuracy_score(polarity_y, y1_result))
    print("[INFO] Precision score for polarity: %s " % precision_score(polarity_y, y1_result, average="macro"))
    print("[INFO] Recall score for polarity: %s " % recall_score(polarity_y, y1_result,average="macro"))
    print("[INFO] F1 score for polarity: %s " % f1_score(polarity_y, y1_result, average="macro"))

    print("[INFO] Accuracy score for purpose: %s " % accuracy_score(purpose_y, y2_result))
    print("[INFO] Precision score for purpose: %s " % precision_score(purpose_y, y2_result, average="macro"))
    print("[INFO] Recall score for purpose: %s " % recall_score(purpose_y, y2_result,average="macro"))
    print("[INFO] F1 score for purpose: %s " % f1_score(purpose_y, y2_result, average="macro"))



    print("-------------------------------Baseline Random-----------------------------")
    y1_result = []
    for i in polarity_y:
        y1_result.append(random.randint(1,3))
    print("y1_result %s" % y1_result)

    y2_result = []
    for i in purpose_y:
        y2_result.append(random.randint(1,6))
    print("y2_result %s" % y2_result)

    y1_result = np.asarray(y1_result)
    y2_result = np.asarray(y2_result)

    print("-------------------------------Evaluation Random-----------------------------")
    print("[INFO] Accuracy score for polarity: %s " % accuracy_score(polarity_y, y1_result))
    print("[INFO] Precision score for polarity: %s " % precision_score(polarity_y, y1_result, average="macro"))
    print("[INFO] Recall score for polarity: %s " % recall_score(polarity_y, y1_result, average="macro"))
    print("[INFO] F1 score for polarity: %s " % f1_score(polarity_y, y1_result, average="macro"))

    print("[INFO] Accuracy score for purpose: %s " % accuracy_score(purpose_y, y2_result))
    print("[INFO] Precision score for purpose: %s " % precision_score(purpose_y, y2_result, average="macro"))
    print("[INFO] Recall score for purpose: %s " % recall_score(purpose_y, y2_result, average="macro"))
    print("[INFO] F1 score for purpose: %s " % f1_score(purpose_y, y2_result, average="macro"))




if __name__ == "__main__":
    print("[INFO] Pipeline started")
    start_time = time.time()
    main()
    print("[INFO] Total processing time: %s seconds" % (time.time() - start_time))