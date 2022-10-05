import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

#See the note on line 29 re: this import.
from helperFunctions.cmPrint import print_cm

ccdb = pd.read_csv("./data/UCI_Credit_Card.csv")

X = ccdb.drop("default.payment.next.month", axis=1)
y = ccdb["default.payment.next.month"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=1693)

scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X. transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=500)
mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)
print(pred)

nn_cm = confusion_matrix(y_test, pred)
print(nn_cm)

#Note I am including the cmPrint function from class in the folder
#"helperFunctions".  I created a file in that folder called "cmPrint",
#which has the print_cm function defined within it.
#See also the import on line 8.
print_cm(nn_cm, ["No Default", "Default"])

updated_mlp = MLPClassifier(hidden_layer_sizes=(10,20,10),
                            activation='tanh',
                            solver="sgd",
                            batch_size=200,
                            learning_rate="adaptive",
                            max_iter=500,
                            verbose=True
                            )

updated_mlp.fit(X_train, y_train)

pred = updated_mlp.predict(X_test)
nn_cm = confusion_matrix(y_test, pred)
print_cm(nn_cm, ["No Default", "Default"])