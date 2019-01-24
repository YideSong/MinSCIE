import time
import csv
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def main():
    texts = [[]]
    purpose =[]
    polarities = []

    # import data
    i =0
    with open("./example.csv") as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            texts[i][0].append(row[3])
            texts[i][1].append(row[5])
            texts[i][2].append(row[7])
            texts[i][3].append(row[9])
            purpose.append(row[11])
            polarities.append(row[12])
            i= i+1
        print(texts[0])

    for line in open("./annotated_sentences.csv"):
        csv_row = line.split()
        #print(line)

        #parts = line.split('\t')
        #if parts[12].strip() != "0":
            #texts.append(parts[5])
            #polarities.append(parts[12].strip())
    #print("[INFO] Imported %s citation contexts and %s polarities." % (len(texts), len(polarities)))
    #print("[INFO] Example context:\n %s" % (texts[0]))
    #print("[INFO] Has a polarity value of %s" % (polarities[0]))
    #print(set(polarities))






if __name__ == "__main__":
    print("[INFO] Pipeline started")
    start_time = time.time()
    main()
    print("[INFO] Total processing time: %s seconds" % (time.time() - start_time))