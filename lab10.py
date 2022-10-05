import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from helperFunctions.cmPrint import print_cm
from helperFunctions.findCorr import find_correlation
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")

ks_data = pd.read_csv("./data/ksprojects201801.csv")

X = ks_data.drop(["ID", "name", "pledged", "backers", "usd pledged", "usd_pledged_real", "state", "category"], axis=1)

y = ks_data["state"]

y[y != "successful"] = "Failed"
y = pd.get_dummies(y)
y = y.drop("Failed", axis=1)
#print(y)
start_date = pd.to_datetime("2000-1-1")

X["deadline"] = (pd.to_datetime(X["deadline"]) - start_date).dt.days
#print(X["deadline"])
X["launched"] = (pd.to_datetime(X["launched"]) - start_date).dt.days
X["duration"] = X["deadline"] - X["launched"]
#print(X["duration"])

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1693)

print("====PCA====")

pca = PCA(n_components = 5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(pca.explained_variance_ratio_)

plt.plot(pca.explained_variance_ratio_)
plt.xticks([0,1,2,3,4])
plt.ylabel("PCA Variance Explained")
plt.savefig("./outputs/PCA_explained_variance.png")
plt.close()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1693)
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


plt.plot(pca.explained_variance_ratio_)
plt.xticks([0,1])
plt.ylabel("PCA Variance Explained")
plt.savefig("./outputs/PCA_explained_variance_2.png")
plt.close()

print("Model with N_components = 2 PCA")
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train_pca, y_train)
y_pred = bayes_classifier.predict(X_test_pca)

confMat = confusion_matrix(y_test, y_pred)
print_cm(confMat, ["Successful", "Failed"])
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred)))

print("Model without PCA (All data columns)")
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train, y_train)
y_pred = bayes_classifier.predict(X_test)

confMat = confusion_matrix(y_test, y_pred)
print_cm(confMat, ["Successful", "Failed"])
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred)))

print("====Kernel PCA All Components ====")
ks_data = ks_data.sample(2500, random_state=1693)
X = ks_data.drop(["ID", "name", "pledged", "backers", "usd pledged", "usd_pledged_real", "state", "category"], axis=1)
y = ks_data["state"]
y[y != "successful"] = "Failed"
y = pd.get_dummies(y)
y = y.drop("Failed", axis=1)
start_date = pd.to_datetime("2000-1-1")
X["deadline"] = (pd.to_datetime(X["deadline"]) - start_date).dt.days
X["launched"] = (pd.to_datetime(X["launched"]) - start_date).dt.days
X["duration"] = X["deadline"] - X["launched"]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1693)

kernel_pca = KernelPCA(kernel="rbf")
X_train_kPCA = kernel_pca.fit_transform(X_train)
X_test_kPCA = kernel_pca.transform(X_test)

explained_variance = np.var(X_train_kPCA, axis=0)
kernelPCA_explained_variance_ratio = explained_variance / np.sum(explained_variance)
print(kernelPCA_explained_variance_ratio)
plt.plot(kernelPCA_explained_variance_ratio)
plt.ylabel("Kernel PCA Variance Explained")
plt.savefig("./outputs/Kernel_PCA_explained_variance.png")
plt.close()

print("====Kernel PCA 250 Components ====")
kernel_pca = KernelPCA(kernel="rbf", n_components=250)
X_train_kPCA = kernel_pca.fit_transform(X_train)
X_test_kPCA = kernel_pca.transform(X_test)

bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train_kPCA, y_train)
y_pred_kPCA = bayes_classifier.predict(X_test_kPCA)

confMat = confusion_matrix(y_test, y_pred_kPCA)
print_cm(confMat, ["Successful", "Failed"])
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred_kPCA)))

print("====Kernel PCA 3 Components ====")
kernel_pca = KernelPCA(kernel="rbf", n_components=3)
X_train_kPCA = kernel_pca.fit_transform(X_train)
X_test_kPCA = kernel_pca.transform(X_test)

bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train_kPCA, y_train)
y_pred_kPCA = bayes_classifier.predict(X_test_kPCA)

confMat = confusion_matrix(y_test, y_pred_kPCA)
print_cm(confMat, ["Successful", "Failed"])
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred_kPCA)))

print("====Finding Correlations in our Data====")
ks_data = pd.read_csv("./data/ksprojects201801.csv")
X = ks_data.drop(["ID", "name", "pledged", "backers", "usd pledged", "usd_pledged_real", "state", "category"], axis=1)
y = ks_data["state"]
y[y != "successful"] = "Failed"
y = pd.get_dummies(y)
y = y.drop("Failed", axis=1)
start_date = pd.to_datetime("2000-1-1")
X["deadline"] = (pd.to_datetime(X["deadline"]) - start_date).dt.days
X["launched"] = (pd.to_datetime(X["launched"]) - start_date).dt.days
X["duration"] = X["deadline"] - X["launched"]
X = pd.get_dummies(X)

print(find_correlation(X, threshold=0.7, remove_negative=True))
X = X.drop(find_correlation(X, threshold=0.7, remove_negative=True), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1693)

bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train, y_train)
y_pred_FC = bayes_classifier.predict(X_test)
confMat = confusion_matrix(y_test, y_pred_FC)
print_cm(confMat, ["Successful", "Failed"])
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred_FC)))