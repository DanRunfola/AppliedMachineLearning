from mpi4py import MPI
import pandas as pd
import random
import math

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

if(MPI.COMM_WORLD.Get_rank() == 0):
    print("I am the rank 0 node.")
    
    #I can send anything I want to other processes - for example:
    param_list = []
    for i in range(0,1000):
        C = random.random() * 10
        param_list.append(C)
    
    #divide our list by the number of cores we have available.
    #We'll round down
    tasks_per_core = math.ceil(len(param_list) / comm.Get_size())
    chunks = [param_list[x:x+tasks_per_core] for x in range(0, len(param_list), 42)]

    for coreID in range(1,comm.Get_size()):
        comm.send(chunks[coreID-1], dest=coreID, tag=11)


if(MPI.COMM_WORLD.Get_rank() != 0):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from time import time

    data = pd.read_csv('studentpor_bigger.csv')

    X = data[["traveltime", "studytime"]].values
    y = data["Walc"]

    scale_X = StandardScaler()
    X = scale_X.fit_transform(X)
    
    parameters = comm.recv(source=0, tag=11)
    print("I (" + str(MPI.COMM_WORLD.Get_rank()) + ") have received " + str(len(parameters)) + " parameters to test.")

    for p in parameters:
        logC = LogisticRegression(penalty="elasticnet", solver="saga", fit_intercept=False, tol=1.0, C=p)
        logC.fit(X, y)

        #Percent accuracy:
        acc = accuracy_score(y, logC.predict(X))
        f = open("/sciclone/home20/dsmillerrunfol/dml/results/" + str(p)+ ".csv", "w")
        f.write(str(p) + "," + str(acc) + "\n")
        f.close()

#Once we're done, we should have a folder full of csv
#files.  All we need to do is concatenate them together
#into one output: cat *csv > all.csv