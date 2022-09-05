from dis import dis
from tkinter import Label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

students_math = pd.read_csv("./data/studentmat.csv")
students_port = pd.read_csv("./data/studentpor.csv")

all_student_rows = [students_math,students_port]
all_students = pd.concat(all_student_rows, ignore_index=True)
print(all_students.shape)

X = all_students[["age", "address", "traveltime", "failures", "higher", "internet", "romantic", "famrel", "freetime", "goout", "absences"]].values

discreteCoder_X = LabelEncoder()

X[:,1] = discreteCoder_X.fit_transform(X[:,1])
X[:,4] = discreteCoder_X.fit_transform(X[:,4])
X[:,5] = discreteCoder_X.fit_transform(X[:,5])
X[:,6] = discreteCoder_X.fit_transform(X[:,6])

y = all_students[["Walc"]].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=1693)

scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X. transform(X_test)

svr_regression = SVR(kernel = "linear", epsilon = 1.0)
svr_regression.fit(X_train, y_train)

#Student A
#Age: 18
#Address: Urban (label encoded as 1)
#Travel Time: 3 (30 minutes to 1 hour)
#Failures: 3
#Desire for Higher Education: No (0)
#Internet Access: No (0)
#Romantica Relationship: Yes (1)
#Relationship with Family: Ok (2 out of 5)
#Freetime: A lot (5 out of 5)
#Going Out: A bit (2 out of 5)
#Absences: 5

new_studentA = [[18, 1, 3, 3, 0, 0, 1, 2, 5, 2, 5]]

new_student_scaledA = scale_X.transform(new_studentA)
studentA_prediction = svr_regression.predict(new_student_scaledA)
print(studentA_prediction)

new_studentB = [[18, 0, 3, 3, 0, 0, 1, 2, 1, 1, 5]]
new_student_scaledB = scale_X.transform(new_studentB)
studentB_prediction = svr_regression.predict(new_student_scaledB)
print(studentB_prediction)

DT_regression = tree.DecisionTreeRegressor(random_state=1693, max_depth=3)
DT_regression.fit(X_train, y_train)

tree.export_graphviz(DT_regression, out_file="outputs/tree.dot", feature_names=["age", "address", "traveltime", "failures", "higher", "internet", "romantic", "famrel", "freetime", "goout", "absences"])

studentA_prediction_RT = DT_regression.predict(new_student_scaledA)
print(studentA_prediction_RT)

studentB_prediction_RT = DT_regression.predict(new_student_scaledB)
print(studentB_prediction_RT)

RF_regression = RandomForestRegressor(n_estimators = 100, random_state=1693)
RF_regression.fit(X_train, y_train)

studentA_prediction_RT = RF_regression.predict(new_student_scaledA)
print(studentA_prediction_RT)

studentB_prediction_RT = RF_regression.predict(new_student_scaledB)
print(studentB_prediction_RT)

rf_MAD = mean_absolute_error(y_test, RF_regression.predict(X_test))
print(rf_MAD)

RT_MAD = mean_absolute_error(y_test, DT_regression.predict(X_test))
SVR_MAD = mean_absolute_error(y_test, svr_regression.predict(X_test))

print(RT_MAD)
print(SVR_MAD)