#Note in the videos I show how to manually install NLTK.
#Times have changed, you can now:
#conda install nltk

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from nltk import download
download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords



wine_data = pd.read_csv("./data/winemagdata130kv2.csv", quoting=2)
#print(wine_data.shape)
#print(wine_data.head(1))

wine_data = wine_data[["description", "points"]]
#print(wine_data.head(1))

wine_data = wine_data.sample(1000, random_state=1693).reset_index(drop=True)
#print(wine_data.shape)

#print(wine_data["description"][0])

wine_description = re.sub('[^a-zA-Z0-9 ]', '', wine_data["description"][0])

#print(wine_description)

wine_description = wine_description.lower()

#print(wine_description)

wine_description = wine_description.split()

#print(wine_description)

#print(stopwords.words('english'))

wine_description = [word for word in wine_description if not word in set(stopwords.words('english'))]

#print(wine_description)

stemmer = PorterStemmer()
wine_description = [stemmer.stem(word) for word in wine_description]

#print(wine_description)

wine_description = " ".join(wine_description)

#print(wine_description)

#Applying to the full dataframe - timing is slightly different than in the video because 
#we aren't in a jupyter notebook.
start = time.time()

corpus = []
for i in range(0, len(wine_data)):
    wine_description = re.sub('[^a-zA-Z0-9 ]', '', wine_data["description"][i])
    wine_description = wine_description.lower()
    wine_description = wine_description.split()
    wine_description = [word for word in wine_description if not word in set(stopwords.words('english'))]
    stemmer = PorterStemmer()
    wine_description = [stemmer.stem(word) for word in wine_description]
    wine_description = " ".join(wine_description)
    corpus.append(wine_description)

end = time.time()
print("CPU time (in seconds): " + str(end-start))

#print(wine_data["description"][2])
#print(corpus[2])

countVec = CountVectorizer()
X_raw = countVec.fit_transform(corpus)
X = X_raw.toarray()

#print(pd.DataFrame(X_raw.A, columns=countVec.get_feature_names()).transpose())

#Note a small difference between the video and this code:
#the current version of matplotlib uses "density=1", not "normed=1".
n, bins, patches = plt.hist(wine_data["points"].values, 10, density=1, facecolor="green", alpha=0.7)
plt.savefig("outputs/nlp_hist.png")


y = wine_data["points"]
y = y.where(y>90, other=0).where(y <= 90, other=1).values
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=1693)
classifier = LogisticRegression(random_state = 1693)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#print(y_pred)

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

# This is slightly different than the video, which has a small error.
confMat = confusion_matrix(y_test, y_pred)
print_cm(confMat, ["Bad Wine", "Good Wine"])