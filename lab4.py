import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

data = pd.read_csv("./data/data.csv")

X = data.iloc[:, 2:4].values
y = data.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=1693)
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X. transform(X_test)

logistic_classifer = LogisticRegression(random_state=1693)
logistic_classifer.fit(X_train, y_train)

y_pred = logistic_classifer.predict(X_test)
y_pred_probabilities = logistic_classifer.predict_proba(X_test)

confMat = confusion_matrix(y_test, y_pred)
print(confMat)

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

print_cm(confMat, ["Benign", "Malignant"])

#KNN
nn_classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
nn_classifier.fit(X_train, y_train)

y_pred_NN = nn_classifier.predict(X_test)
confMat_NN = confusion_matrix(y_test, y_pred_NN)
print_cm(confMat_NN, ["Benign", "Malignant"])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder

def viz_cm(model, labels, fPath="outputs/vizCmOutput.png"):

    X_set, y_set = X_train, y_train
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

#Note this will simply output to outputs/vizCmOutput.png by default.
#You can change it to output to a jupyter notebook by uncommenting the plt.show()
#line in the function.  You can pass the options fPath= option
#to this function to change the destination of the graphical output.
viz_cm(model=nn_classifier, labels=["Radius Mean", "Texture Mean"])


#Support Vector
svc_classifier = SVC(kernel="linear", random_state=1693)
svc_classifier.fit(X_train, y_train)

y_pred_SVC = svc_classifier.predict(X_test)
confMat_SVC = confusion_matrix(y_test, y_pred_SVC)
print_cm(confMat_SVC, ["Benign", "Malignant"])
viz_cm(model=svc_classifier, labels=["Radius Mean", "Texture Mean"], fPath="outputs/svcClass.png")


#Kernel SVM
kernelSVC_classifier = SVC(kernel="rbf", random_state=1693)
kernelSVC_classifier.fit(X_train, y_train)

y_pred_SVC_kernel = kernelSVC_classifier.predict(X_test)
confMat_SVC_kernel = confusion_matrix(y_test, y_pred_SVC_kernel)
print_cm(confMat_SVC_kernel, ["Benign", "Malignant"])
viz_cm(model=kernelSVC_classifier, labels=["Radius Mean", "Texture Mean"], fPath="outputs/kernel_svcClass.png")