import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("data/data.csv")

X = data.iloc[:,2:4].values
y = data.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=1693)
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X. transform(X_test)

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

def viz_cm(model, labels, fPath="outputs/vizCmOutput.png", set = "train"):

    if(set == "train"):
        X_set, y_set = X_train, y_train
    else:
        X_set, y_set = X_test, y_test

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    pred = model.predict(np.array([X1.ravel(), X2.ravel()]).T)

    discreteCoder = LabelEncoder()
    pred = discreteCoder.fit_transform(pred)
    plt.figure()
    plt.contourf(X1, X2, pred.reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Classification')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    
    #If you're on Jupyter Notebook, you can use this:
    #plt.show()
    plt.savefig(fPath)


bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train, y_train)
y_pred = bayes_classifier.predict(X_test)

confMat = confusion_matrix(y_test, y_pred)
print_cm(confMat, ["Benign", "Malignant"])

viz_cm(bayes_classifier, ["Radius Mean", "Texture Mean"], fPath="outputs/bayes_class_train.png")
viz_cm(bayes_classifier, ["Radius Mean", "Texture Mean"], fPath="outputs/bayes_class_test.png", set="test")


dt_classifier = DecisionTreeClassifier(random_state = 1693, max_depth=3)
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)

dt_confMat = confusion_matrix(y_test, dt_pred)
print_cm(dt_confMat, ["Benign", "Malignant"])

viz_cm(dt_classifier, ["Radius Mean", "Texture Mean"], fPath="outputs/dt_class_train.png")

tree.export_graphviz(dt_classifier, out_file="tree_dt.dot", feature_names=["Radius Mean", "Texture Mean"])

rt_classifier = RandomForestClassifier(n_estimators=1000, random_state=1693, max_depth=3)
rt_classifier.fit(X_train, y_train)
rt_pred = rt_classifier.predict(X_test)
rt_cm = confusion_matrix(y_test, rt_pred)
print_cm(rt_cm, ["Benign", "Malignant"])

viz_cm(rt_classifier, ["Radius Mean", "Texture Mean"], fPath="outputs/rf_class_train.png")