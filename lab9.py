from curses.ascii import GS
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

dta = pd.read_csv("./data/dailyinmatesincustody.csv")

dta = dta.dropna(subset=["GENDER"])

X = dta[["AGE", "GENDER"]]
y = dta["INFRACTION"]

X = pd.get_dummies(X)
X = X.drop(columns=["GENDER_M"])

y = pd.get_dummies(y)
y = y.drop(columns=["N"])

X = X.values
y = y.values

dt_classifier = DecisionTreeClassifier(random_state=1693, max_depth=2, min_samples_leaf=2)
dt_classifier.fit(X,y)

k_fold = cross_val_score(estimator=dt_classifier, X=X, y=y, cv=10, scoring="accuracy")

print(k_fold)
print(k_fold.mean())

params =  [{"max_depth":[1,2,3,4,5], 
            "min_samples_leaf":[2,4,5,6,10,12,14,16,18,20]  
          }]

gSearch = GridSearchCV(estimator=dt_classifier,
                       param_grid=params,
                       scoring='accuracy',
                       cv=10 
                      )

gSearch_results = gSearch.fit(X,y)

print(gSearch_results)
print(gSearch_results.best_params_)
print(gSearch_results.best_score_)
