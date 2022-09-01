import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

data = pd.read_csv('studentpor.csv')

#We'll run a model that attempts to predict if a student
#if a student has very low (0) or very high (5) weekend
#alcohol consumption (Walc) based on their studytime weekly
#(studytime) and traveltime to school (traveltime).
#For this example, we'll be treating studytime
#and traveltime as continious variables, which isn't
#perfectly accurate, but close enough. We aren't really
#trying to get the best model we can, but rather illustrate
#a distribution strategy.  We're also not doing lots of other
#things we might want to, like split our data, etc.

X = data[["traveltime", "studytime"]].values
y = data["Walc"]

scale_X = StandardScaler()
X = scale_X.fit_transform(X)

#C is our regularization strength - larger values mean
#weaker regularization.
#We'll do a random search here, from 0 to 10.
C = random.random() * 10
logC = LogisticRegression(penalty="elasticnet", solver="saga", fit_intercept=False, tol=1.0, C=1.0)
logC.fit(X, y)

#Percent accuracy:
acc = accuracy_score(y, logC.predict(X))

#Save it into a file with our C:
f = open("/sciclone/home20/dsmillerrunfol/dml/results/" + str(C)+ ".csv", "w")
f.write(str(C) + "," + str(acc) + "\n")
f.close()

#Once we're done, we should have a folder full of csv
#files.  All we need to do is concatenate them together
#into one output: cat *csv > all.csv