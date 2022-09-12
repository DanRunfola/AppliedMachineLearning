import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from time import time

#This is the same as the dataset we used before, but
#about an order of magnitude bigger due to copying rows.
#This is to illustrate how these methods perform using
#reasonably large datasets.
data = pd.read_csv('studentpor_bigger.csv')

#Similar to the original grid search example,
#Here we'll be implementing a parallel job across
#a single node by increasing the number of threads
#we use.  We'll time things to show the difference
#between single and multi-core use.
#Note in the job script the -c 1 command!
#That tells the scheduler to only create *one*
#version of this script.  This script then handles
#the parallelization itself.


X = data[["traveltime", "studytime"]].values
y = data["Walc"]

scale_X = StandardScaler()
X = scale_X.fit_transform(X)


#This will be slow - one job only.
start = time()
rf = RandomForestClassifier(n_estimators=500, n_jobs=1)
rf.fit(X, y)
end = time()
res = end-start
print('%.3f seconds (1 core)' % res)
#For my run, n_jobs = 1 results in a 27.131 second runtime.


#Now we'll set this to 12 - this will create 12 processes.
start = time()
rf = RandomForestClassifier(n_estimators=500, n_jobs=12)
rf.fit(X, y)
end = time()
res = end-start
print('%.3f seconds (12 cores)' % res)
#For my run, this results in a 8.334 second runtime.
