import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sin,pi
import random

def Generate_Synthetic_Dataset(size):
    Xsize = size
    start = 0
    end   = 1
    Noise = np.random.normal(0,0.3, Xsize)                      # np.random.normal(Mean, Sd, Size)
    X = np.linspace(start, end, num=Xsize)                      # np.linspace(Start, End, Size)
    Y = np.sin(2*np.pi*X)
    Y = Y+Noise
    df = pd.DataFrame()
    df['X'] = pd.Series(X)
    df['Target'] = pd.Series(Y)
    df.to_csv("Dataset.csv", sep='\t', encoding='utf-8')
    return X,Y

def Split_Train_Test(X,Y):
    Xsize = len(X)
    Shuffler = np.linspace(0,Xsize-1,Xsize)
    random.shuffle(Shuffler)
    TrainX = []
    TrainY = []
    TestX = []
    TestY = []
    for i in range(0,Xsize):
        if i < int(0.8*Xsize):
            TrainX.append(X[int(Shuffler[i])])
            TrainY.append(Y[int(Shuffler[i])])
        else:
            TestX.append(X[int(Shuffler[i])])
            TestY.append(Y[int(Shuffler[i])])

    TrainX = np.array(TrainX)
    TrainY = np.array(TrainY)
    TestX = np.array(TestX)
    TestY = np.array(TestY)
    return TrainX, TrainY, TestX, TestY


def gradient_descent(X, Y, alpha, degree, iteration):
    m = len(X)

    # Parameters
    Thita = np.zeros(degree+1)
    for itr in range(0,iteration):

        # Hypothesis
        H = np.zeros(m)
        for i in range(0,degree+1):
            H = H+ Thita[i]*(np.power(X,i))
            
        # Cost
        Cost = (1.0/(2.0*m))*( sum( (H-Y)*(H-Y) ) )
        
        # Gradient Descent
        for i in range(0,degree+1):
            Temp = (H-Y)*(np.power(X,i))
            Thita[i] = Thita[i] - (alpha*(1.0/m)*(sum(Temp)))
            
    return Thita

def PredictError(X, Y, Thita):
    m = len(X)
    degree = len(Thita)-1
    H = np.zeros(m)
    for i in range(0,degree+1):
        H = H+ Thita[i]*(np.power(X,i))
    Cost = (1.0/(2.0*m))*( sum( (H-Y)*(H-Y) ) )
    return Cost

def DataFrame_Linear_Regression(TrainX, TrainY, TestX, TestY, No_of_iteration):
    DF_Thita = pd.DataFrame()
    DF_Error = pd.DataFrame()
    TestError =[]
    
    # Varing N from 1 to 9
    for N in range(9,0,-1):
    	# Estimating the Parameters
        Thita = gradient_descent(TrainX, TrainY, 0.05, N, No_of_iteration)

        # Storing the Thita in a Dataframe
        DF_Thita['N='+str(N)]=pd.Series(Thita)
        print "Squared Error in Test Set when N=",N, " is:", PredictError(TestX,TestY,Thita)
        
        TestError.append(PredictError(TestX,TestY,Thita))

    TestError = TestError[::-1]
    TestError = np.array(TestError)
    
    for i in range(9,0,-1):
        DF_Thita['N='+str(i)].fillna(' ',inplace=True)
        DF_Error['Test Error']=pd.Series(TestError)
    DF_Thita.rename(index={0:'Thita0',1:'Thita1',2:'Thita2',3:'Thita3',4:'Thita4',5:'Thita5',6:'Thita6',7:'Thita7',8:'Thita8',9:'Thita9'},inplace=True)
    DF_Error.rename(index={0:'N=1',1:'N=2',2:'N=3',3:'N=4',4:'N=5',5:'N=6',6:'N=7',7:'N=8',8:'N=9'},inplace=True)
    return DF_Thita, DF_Error


# Main Function:
No_of_iteration = 10000
SIZE = 10
X,Y = Generate_Synthetic_Dataset(SIZE)
TrainX, TrainY, TestX, TestY = Split_Train_Test(X,Y)
DF_Thita10, DF_Error10 = DataFrame_Linear_Regression(TrainX, TrainY, TestX, TestY, No_of_iteration)
DF_Thita10.to_csv("Parameters.csv", sep='\t', encoding='utf-8')
DF_Error10.to_csv("Test_Error.csv", sep='\t', encoding='utf-8')