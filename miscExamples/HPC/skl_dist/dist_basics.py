from mpi4py import MPI
import pandas as pd
import random

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

#Note that comm.Get_size() is going to be based on the number of cores you have in your job script.
#Here it should be 24 (2 nodes, 12 cores each)
print("Hello from " + str(comm.Get_rank()) +".  There are a total of " + str(comm.Get_size()) + " of us.  Good luck.")

if(MPI.COMM_WORLD.Get_rank() == 0):
    print("I am the rank 0 node.  Code you write here will only execute on this process.")
    print("This is frequently used to create a master node that collects results from other nodes.")
    
    #I can send anything I want to other processes - for example:
    param_list = []
    for i in range(0,1000):
        C = random.random() * 10
        param_list.append(C)
    comm.send(param_list[0:100], dest=1, tag=11)

if(MPI.COMM_WORLD.Get_rank() ==1):
    #I want to load the data on individual nodes - 
    #not send it over the network, as that is much slower.
    data = pd.read_csv('studentpor_bigger.csv')
    parameters = comm.recv(source=0, tag=11)
    print(parameters)

