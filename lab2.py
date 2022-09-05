from random import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

piazza_data = pd.read_csv("./data/piazzadata201_statsSpring18.csv")

#Just as a quick example...
with open("outputs/gridVariableNames.txt","wt") as f:
    print(piazza_data.columns.tolist(), file=f)

X = piazza_data[["contributions"]].values
y = piazza_data[["Grade"]].values



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=1693)

regression = LinearRegression()
regression.fit(X_train, y_train)

#Savefig for python - you can do plt.show() in jupyter
plt.scatter(X_train, y_train)
plt.savefig("outputs/scatterA.png")


#Second figure
plt.scatter(X_train, y_train, color="black")
plt.plot(X_train, regression.predict(X_train), color="red")
plt.savefig("outputs/scatterPred.png")

#Third
plt.plot(X_train, regression.predict(X_train), color="red")
plt.title("Piazza contributions and Grades (Training Data)")
plt.xlabel("Piazza contributions")
plt.ylabel("Grade")
plt.savefig("outputs/scatterPredPretty.png")


y_predictions = regression.predict(X_test)
plt.scatter(X_train, y_train, color="black")
plt.scatter(X_test, y_test, color="blue")
plt.plot(X_train, regression.predict(X_train), color="red")
plt.title("Piazza contributions and Grades")
plt.xlabel("Piazza contributions")
plt.ylabel("Grade")
plt.savefig("outputs/scatterTestTrain.png")

X = piazza_data[["contributions", "days online", "views", "questions", "answers"]].values
y = piazza_data[["Grade"]].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state = 1693)


scale_X = StandardScaler()
X_train_std = scale_X.fit_transform(X_train)

multiple_regression = LinearRegression()
multiple_regression.fit(X_train_std, y_train)

y_predictions = multiple_regression.predict(X_train_std)

#To find your regression coefficients and intercept manually:
print(multiple_regression.intercept_)
print(multiple_regression.coef_)

#If you need more traditional statistical outputs (i.e., f-stats, AIC, log likelihoods, R2, etc. etc.)
#I recommend using the python package statsmodels.  

#Polynomial...
X = piazza_data[["contributions", "days online", "views", "questions", "answers"]].values
y = piazza_data[["Grade"]].values

poly_data = PolynomialFeatures(degree = 2)
X_poly = poly_data.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly,y,test_size=0.25, random_state = 1693)
poly_reg = LinearRegression()
poly_reg.fit(X_train, y_train)


#Comparison
X = piazza_data[["contributions"]].values
y = piazza_data[["Grade"]].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=1693)

scale_X = StandardScaler()
X_train_std = scale_X.fit_transform(X_train)
X_test_std = scale_X.fit(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

poly_data=PolynomialFeatures(degree=2)
poly_reg= LinearRegression()
poly_reg.fit(poly_data.fit_transform(X_train), y_train)

plt.scatter(X_test, y_test, color="black", label="Truth")
plt.scatter(X_test, lin_reg.predict(X_test), color="green", label="Linear")
plt.scatter(X_test, poly_reg.predict(poly_data.fit_transform(X_test)), color="blue", label="Poly")
plt.xlabel("Piazza Contribution")
plt.ylabel("Grade")
plt.legend()
plt.savefig("outputs/scatterPredPrettyComparison.png")