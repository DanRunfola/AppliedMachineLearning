import random

def playSlots(machine):

       
    if(machine == 1):
        return round(random.uniform(0,1),2)
    
    if(machine == 2):
        return round(random.uniform(0.5,1.5),2)
    
    if(machine>=3 and machine <=10):
        return round(random.triangular(0,0.5,1),2)
    
    if(machine >=11 and machine <=14):
        return round(random.triangular(0.5,1,1.5),2)
    
    if(machine == 15):
        return round(random.triangular(0.8,1,3.0),2)
    
    if(machine >= 16 and machine <= 20):
        return round(random.uniform(0,1.1),2)

    if(machine >20):
        print("There are only 20 slot machines!  You've gone awry.")