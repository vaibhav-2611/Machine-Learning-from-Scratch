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
    df.to_csv('Dataset'+str(size)+'.csv', sep='\t', encoding='utf-8')
    plt.figure(figsize=(12, 6))
    plt.scatter(X,Y)
    plt.ylabel('Target')
    plt.xlabel('X')
    plt.title('Synthetic Dataset Size:'+str(size))
    plt.savefig('Syn_Data_size_'+str(size)+'.png')
    # plt.show()
    plt.close()
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
    Pcost = []
    Pitr  = []
    # Parameters
    Thita = np.zeros(degree+1)
    for itr in range(0,iteration):
        Pitr.append(itr)
        # Hypothesis
        H = np.zeros(m)
        for i in range(0,degree+1):
            H = H+ Thita[i]*(np.power(X,i))
            
        # Cost
        Cost = (1.0/(2.0*m))*( np.sum( ((H-Y)*(H-Y))*((H-Y)*(H-Y)) ) )
        Pcost.append(Cost)
        # Gradient Descent
        for i in range(0,degree+1):
            Temp = ((H-Y)*(H-Y)*(H-Y))*(np.power(X,i))
            Thita[i] = Thita[i] - (alpha*(2.0/m)*(np.sum(Temp)))
    
    Pcost = np.array(Pcost)
    Pitr  = np.array(Pitr)
    plt.title('Learning Curve when n='+str(degree))
    plt.xlabel('No of iterations')
    plt.ylabel('Cost')
    plt.plot(Pitr, Pcost)
    return Thita

def PredictError(X, Y, Thita):
    m = len(X)
    degree = len(Thita)-1
    H = np.zeros(m)
    for i in range(0,degree+1):
        H = H+ Thita[i]*(np.power(X,i))
    Cost = (1.0/(2.0*m))*( np.sum( ((H-Y)*(H-Y))*((H-Y)*(H-Y)) ) )
    return Cost

def Predict(X, Thita):
    degree = len(Thita)-1
    H = np.zeros(len(X))
    for i in range(0,degree+1):
        H = H+ Thita[i]*(np.power(X,i))
    return H

def DataFrame_Linear_Regression(TrainX, TrainY, TestX, TestY, No_of_iteration):
    DF_Thita = pd.DataFrame()
    DF_Error = pd.DataFrame()
    TestError =[]
    TrainError =[]
    # Varing N from 1 to 9
    for N in range(9,0,-1):
        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)
        # Estimating the Parameters
        Thita = gradient_descent(TrainX, TrainY, 0.05, N, No_of_iteration)

        # Plot of Hypothesis, Test Set, Prediction on Test Set, Train Set, Prediction on Train Set
        plt.subplot(1,2,2)
        plt.scatter(TrainX, TrainY, color ='b',marker='*', label= 'Train Set')
        plt.scatter(TestX, TestY, color ='r', marker='+',label= 'Test Set')
        
        ALL = np.linspace(0, 1 , 50)
        PredY = Predict(ALL, Thita)
        plt.plot(ALL, PredY, color='g', linestyle='--', label='Hypothesis' )
        
        PredY = Predict(TrainX, Thita)
        plt.scatter(TrainX, PredY, color='c',  marker='*', label='Prediction on Train Set' )
        
        PredY_test = Predict(TestX, Thita)
        plt.scatter(TestX, PredY_test, color='g',  marker='+', label='Prediction on Test Set' )        
        
        plt.title('n = '+str(N)+'    Learning Rate = 0.05')
        plt.xlabel('X')
        plt.ylabel('Target')
        plt.legend()
        plt.savefig('Data'+str(len(TrainX)+len(TestX))+'_N'+str(N)+'.png')
        # plt.show()
        plt.close()
        
        # Storing the Thita in a Dataframe
        DF_Thita['N='+str(N)]=pd.Series(Thita)
        a = PredictError(TestX,TestY,Thita)
        b = PredictError(TrainX,TrainY,Thita)
        print "Squared Error in Test Set when N=",N, " is:", a
        print "Squared Error in Train Set when N=",N, " is:", b
        TestError.append(a)
        TrainError.append(b)

    TestError = TestError[::-1]
    TrainError = TrainError[::-1]
    TestError = np.array(TestError)
    TrainError = np.array(TrainError)
    
    # plot of train error and test error
    Error_X = np.linspace(1,9,9)
    plt.title("Squared error on both train and test data")
    plt.plot(Error_X, TrainError, color='g', marker='o', linestyle='--', label='Train Error')
    plt.plot(Error_X, TestError,  color='r', marker='o', linestyle='--', label='Test Error' )
    plt.xlabel('N')
    plt.ylabel('Test Error and Train Error')
    plt.legend()
    plt.savefig('Train_Test_Squared_Error'+str(len(TrainX)+len(TestY))+'.png')
    # plt.show()
    plt.close()

    # Return Purpose
    for i in range(9,0,-1):
        DF_Thita['N='+str(i)].fillna(' ',inplace=True)
        DF_Error['Test Error']=pd.Series(TestError)
        DF_Error['Train Error']=pd.Series(TrainError)
    DF_Thita.rename(index={0:'Thita0',1:'Thita1',2:'Thita2',3:'Thita3',4:'Thita4',5:'Thita5',6:'Thita6',7:'Thita7',8:'Thita8',9:'Thita9'},inplace=True)
    DF_Error.rename(index={0:'N=1',1:'N=2',2:'N=3',3:'N=4',4:'N=5',5:'N=6',6:'N=7',7:'N=8',8:'N=9'},inplace=True)
    return DF_Thita, DF_Error

#========================================= PART1 ================================== #
No_of_iteration = 10000
df = pd.DataFrame()

print "When Dataset Size = 10"
SIZE = 10
X,Y = Generate_Synthetic_Dataset(SIZE)
TrainX, TrainY, TestX, TestY = Split_Train_Test(X,Y)
DF_Thita10, DF_Error10 = DataFrame_Linear_Regression(TrainX, TrainY, TestX, TestY, No_of_iteration)
DF_Thita10.to_csv("Parameters_10.csv", sep='\t', encoding='utf-8')
DF_Error10.to_csv("Train_Test_Error_10.csv", sep='\t', encoding='utf-8')
X1=X
Y1=Y

print "When Dataset Size = 100"
SIZE = 100
X,Y = Generate_Synthetic_Dataset(SIZE)
TrainX, TrainY, TestX, TestY = Split_Train_Test(X,Y)
DF_Thita100, DF_Error100 = DataFrame_Linear_Regression(TrainX, TrainY, TestX, TestY, No_of_iteration)
DF_Thita100.to_csv("Parameters_100.csv", sep='\t', encoding='utf-8')
DF_Error100.to_csv("Train_Test_Error_100.csv", sep='\t', encoding='utf-8')

print "When Dataset Size = 100"
SIZE = 100
X,Y = Generate_Synthetic_Dataset(SIZE)
TrainX, TrainY, TestX, TestY = Split_Train_Test(X,Y)
DF_Thita100, DF_Error100 = DataFrame_Linear_Regression(TrainX, TrainY, TestX, TestY, No_of_iteration)
DF_Thita100.to_csv("Parameters_100.csv", sep='\t', encoding='utf-8')
DF_Error100.to_csv("Train_Test_Error_100.csv", sep='\t', encoding='utf-8')

print "When Dataset Size = 1000"
SIZE = 1000
X,Y = Generate_Synthetic_Dataset(SIZE)
TrainX, TrainY, TestX, TestY = Split_Train_Test(X,Y)
DF_Thita1000, DF_Error1000 = DataFrame_Linear_Regression(TrainX, TrainY, TestX, TestY, No_of_iteration)
DF_Thita1000.to_csv("Parameters_1000.csv", sep='\t', encoding='utf-8')
DF_Error1000.to_csv("Train_Test_Error_1000.csv", sep='\t', encoding='utf-8')

print "When Dataset Size = 10000"
SIZE = 10000
X,Y = Generate_Synthetic_Dataset(SIZE)
TrainX, TrainY, TestX, TestY = Split_Train_Test(X,Y)
DF_Thita10000, DF_Error10000 = DataFrame_Linear_Regression(TrainX, TrainY, TestX, TestY, No_of_iteration)
DF_Thita10000.to_csv("Parameters_10000.csv", sep='\t', encoding='utf-8')
DF_Error10000.to_csv("Train_Test_Error_10000.csv", sep='\t', encoding='utf-8')


# ======================================== Part3 ======================================= #
for N in range(1,10):
    LC_X = [10,100,1000,10000]
    LC_train = []
    LC_test  = []

    LC_test.append( DF_Error10.loc['N='+str(N)]['Test Error'] )
    LC_train.append( DF_Error10.loc['N='+str(N)]['Train Error'] )


    LC_test.append( DF_Error100.loc['N='+str(N)]['Test Error'] )
    LC_train.append( DF_Error100.loc['N='+str(N)]['Train Error'] )

    LC_test.append( DF_Error1000.loc['N='+str(N)]['Test Error'] )
    LC_train.append( DF_Error1000.loc['N='+str(N)]['Train Error'] )

    LC_test.append( DF_Error10000.loc['N='+str(N)]['Test Error'] )
    LC_train.append( DF_Error10000.loc['N='+str(N)]['Train Error'] )

    LC_X = np.array(LC_X)
    LC_train = np.array(LC_train)
    LC_test  = np.array(LC_test)
    
    plt.figure(figsize=(10,6))
    plt.loglog(LC_X , LC_train, color='g', marker='o', linestyle='--', label='Train Error')
    plt.loglog(LC_X , LC_test,  color='r', marker='o', linestyle='--', label='Test Error')
    plt.xlabel('Dataset Size ( in Log Scale)')
    plt.ylabel('Error ( in Log Scale)')
    plt.title('Experiment with varying Dataset Size N='+str(N))
    plt.legend()
    plt.savefig('LC_Dataset_size_N'+str(N)+'.png')
    # plt.show()
    plt.close()

# =================================== PART4.a.i =================================== #
def gradient_descent3(X, Y, alpha, degree, iteration):  #Mean Absolute
    m = len(X)
    Pcost = []
    Pitr  = []
    # Parameters
    Thita = np.zeros(degree+1)
    for itr in range(0,iteration):
        Pitr.append(itr)
        # Hypothesis
        H = np.zeros(m)
        for i in range(0,degree+1):
            H = H+ Thita[i]*(np.power(X,i))
            
        # Cost
        Cost = (1.0/(2.0*m))*( sum( abs(H-Y) ) )
        Pcost.append(Cost)
        # Gradient Descent
        t=0
        for i in range(0,degree+1):
            for k in range(len(H-Y)):
                if((H-Y)[k] > 0):
                    t += (np.power(X[k],i))
                else:
                    t -= (np.power(X[k],i))
            Thita[i] = Thita[i] - (alpha*(1.0/(2.0*m))*(t))
    
    Pcost = np.array(Pcost)
    Pitr  = np.array(Pitr)
    plt.title('Learning Curve when n='+str(degree))
    plt.xlabel('No of iterations')
    plt.ylabel('Cost')
    plt.plot(Pitr, Pcost)
    return Thita

def PredictError3(X, Y, Thita):   # Mean Absolute 
    m = len(X)
    degree = len(Thita)-1
    H = np.zeros(m)
    for i in range(0,degree+1):
        H = H+ Thita[i]*(np.power(X,i))
    Cost = (1.0/(2.0*m))*( np.sum( np.abs(H-Y)) )
    return Cost

def DataFrame_Linear_Regression3(TrainX, TrainY, TestX, TestY, No_of_iteration):
    DF_Thita = pd.DataFrame()
    DF_Error = pd.DataFrame()
    TestError =[]
    TrainError =[]
    # Varing N from 1 to 9
    for N in range(9,0,-1):
        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)
        # Estimating the Parameters
        Thita = gradient_descent3(TrainX, TrainY, 0.05, N, No_of_iteration)

        # Plot of Hypothesis, Test Set, Prediction on Test Set, Train Set, Prediction on Train Set
        plt.subplot(1,2,2)
        plt.scatter(TrainX, TrainY, color ='b',marker='*', label= 'Train Set')
        plt.scatter(TestX, TestY, color ='r', marker='+',label= 'Test Set')
        
        ALL = np.linspace(0, 1 , 50)
        PredY = Predict(ALL, Thita)
        plt.plot(ALL, PredY, color='g', linestyle='--', label='Hypothesis' )
        
        PredY = Predict(TrainX, Thita)
        plt.scatter(TrainX, PredY, color='c',  marker='*', label='Prediction on Train Set' )
        
        PredY_test = Predict(TestX, Thita)
        plt.scatter(TestX, PredY_test, color='g',  marker='+', label='Prediction on Test Set' )        
        
        plt.title('n = '+str(N)+'    Learning Rate = 0.05')
        plt.xlabel('X')
        plt.ylabel('Target')
        plt.legend()
        plt.savefig('Data'+str(len(TrainX)+len(TestX))+'_N'+str(N)+'_cost3.png')
        # plt.show()
        plt.close()
        
        # Storing the Thita in a Dataframe
        DF_Thita['N='+str(N)]=pd.Series(Thita)
        a = PredictError3(TestX,TestY,Thita)
        b = PredictError3(TrainX,TrainY,Thita)
        print "Mean absolute Error in Test Set when N=",N, " is:", a
        print "Mean absolute Error in Train Set when N=",N, " is:", b
        TestError.append(a)
        TrainError.append(b)

    TestError = TestError[::-1]
    TrainError = TrainError[::-1]
    TestError = np.array(TestError)
    TrainError = np.array(TrainError)
    
    # plot of train error and test error
    plt.figure()
    Error_X = np.linspace(1,9,9)
    plt.title("Mean absolute error on both train and test data")
    plt.plot(Error_X, TrainError, color='g', marker='o', linestyle='--', label='Train Error')
    plt.plot(Error_X, TestError,  color='r', marker='o', linestyle='--', label='Test Error' )
    plt.xlabel('N')
    plt.ylabel('Test Error and Train Error')
    plt.legend()
    plt.savefig('Train_Test_Mean_absolute_Error_cost3.png')
    # plt.show()
    plt.close()


    # Return Purpose
    for i in range(9,0,-1):
        DF_Thita['N='+str(i)].fillna(' ',inplace=True)
        DF_Error['Test Error']=pd.Series(TestError)
        DF_Error['Train Error']=pd.Series(TrainError)
    DF_Thita.rename(index={0:'Thita0',1:'Thita1',2:'Thita2',3:'Thita3',4:'Thita4',5:'Thita5',6:'Thita6',7:'Thita7',8:'Thita8',9:'Thita9'},inplace=True)
    DF_Error.rename(index={0:'N=1',1:'N=2',2:'N=3',3:'N=4',4:'N=5',5:'N=6',6:'N=7',7:'N=8',8:'N=9'},inplace=True)
    return DF_Thita, DF_Error

SIZE = 10
TrainX1, TrainY1, TestX1, TestY1 = Split_Train_Test(X1,Y1)
DF_Thita10cost3, DF_Error10cost3 = DataFrame_Linear_Regression3(TrainX1, TrainY1, TestX1, TestY1, No_of_iteration)
DF_Thita10cost3.to_csv("Parameters_10_cost3.csv", sep='\t', encoding='utf-8')
DF_Error10cost3.to_csv("Train_Test_Error_10_cost3.csv", sep='\t', encoding='utf-8')

# =================================== PART4.a.ii =================================== #
def gradient_descent2(X, Y, alpha, degree, iteration):   #Fourth power
    m = len(X)
    Pcost = []
    Pitr  = []
    # Parameters
    Thita = np.zeros(degree+1)
    for itr in range(0,iteration):
        Pitr.append(itr)
        # Hypothesis
        H = np.zeros(m)
        for i in range(0,degree+1):
            H = H+ Thita[i]*(np.power(X,i))
            
        # Cost
        Cost = (1.0/(2.0*m))*( sum( ((H-Y)*(H-Y))*((H-Y)*(H-Y)) ) )
        Pcost.append(Cost)
        # Gradient Descent
        for i in range(0,degree+1):
            Temp = ((H-Y)*(H-Y)*(H-Y))*(np.power(X,i))
            Thita[i] = Thita[i] - (alpha*(2.0/m)*(sum(Temp)))
    
    Pcost = np.array(Pcost)
    Pitr  = np.array(Pitr)
    plt.title('Learning Curve when n='+str(degree))
    plt.xlabel('No of iterations')
    plt.ylabel('Cost')
    plt.plot(Pitr, Pcost)
    return Thita
def PredictError2(X, Y, Thita):  #Fourth power
    m = len(X)
    degree = len(Thita)-1
    H = np.zeros(m)
    for i in range(0,degree+1):
        H = H+ Thita[i]*(np.power(X,i))
    Cost = (1.0/(2.0*m))*( sum( ((H-Y)*(H-Y))*((H-Y)*(H-Y)) ) )
    return Cost

def DataFrame_Linear_Regression2(TrainX, TrainY, TestX, TestY, No_of_iteration):
    DF_Thita = pd.DataFrame()
    DF_Error = pd.DataFrame()
    TestError =[]
    TrainError =[]
    # Varing N from 1 to 9
    for N in range(9,0,-1):
        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)
        # Estimating the Parameters
        Thita = gradient_descent2(TrainX, TrainY, 0.05, N, No_of_iteration)

        # Plot of Hypothesis, Test Set, Prediction on Test Set, Train Set, Prediction on Train Set
        plt.subplot(1,2,2)
        plt.scatter(TrainX, TrainY, color ='b',marker='*', label= 'Train Set')
        plt.scatter(TestX, TestY, color ='r', marker='+',label= 'Test Set')
        
        ALL = np.linspace(0, 1 , 50)
        PredY = Predict(ALL, Thita)
        plt.plot(ALL, PredY, color='g', linestyle='--', label='Hypothesis' )
        
        PredY = Predict(TrainX, Thita)
        plt.scatter(TrainX, PredY, color='c',  marker='*', label='Prediction on Train Set' )
        
        PredY_test = Predict(TestX, Thita)
        plt.scatter(TestX, PredY_test, color='g',  marker='+', label='Prediction on Test Set' )        
        
        plt.title('n = '+str(N)+'    Learning Rate = 0.05')
        plt.xlabel('X')
        plt.ylabel('Target')
        plt.legend()
        plt.savefig('Data'+str(len(TrainX)+len(TestX))+'_N'+str(N)+'_cost2.png')
        # plt.show()
        plt.close()

        # Storing the Thita in a Dataframe
        DF_Thita['N='+str(N)]=pd.Series(Thita)
        a = PredictError2(TestX,TestY,Thita)
        b = PredictError2(TrainX,TrainY,Thita)
        print "Four power Error in Test Set when N=",N, " is:", a
        print "Four power Error in Train Set when N=",N, " is:", b
        TestError.append(a)
        TrainError.append(b)

    TestError = TestError[::-1]
    TrainError = TrainError[::-1]
    TestError = np.array(TestError)
    TrainError = np.array(TrainError)
    
    # plot of train error and test error
    plt.figure()
    Error_X = np.linspace(1,9,9)
    plt.title("Four power error on both train and test data")
    plt.plot(Error_X, TrainError, color='g', marker='o', linestyle='--', label='Train Error')
    plt.plot(Error_X, TestError,  color='r', marker='o', linestyle='--', label='Test Error' )
    plt.xlabel('N')
    plt.ylabel('Test Error and Train Error')
    plt.legend()
    plt.savefig('Train_Test_Four_power_Error_cost2.png')
    # plt.show()
    plt.close()

    # Return Purpose
    for i in range(9,0,-1):
        DF_Thita['N='+str(i)].fillna(' ',inplace=True)
        DF_Error['Test Error']=pd.Series(TestError)
        DF_Error['Train Error']=pd.Series(TrainError)
    DF_Thita.rename(index={0:'Thita0',1:'Thita1',2:'Thita2',3:'Thita3',4:'Thita4',5:'Thita5',6:'Thita6',7:'Thita7',8:'Thita8',9:'Thita9'},inplace=True)
    DF_Error.rename(index={0:'N=1',1:'N=2',2:'N=3',3:'N=4',4:'N=5',5:'N=6',6:'N=7',7:'N=8',8:'N=9'},inplace=True)
    return DF_Thita, DF_Error

SIZE = 10
TrainX1, TrainY1, TestX1, TestY1 = Split_Train_Test(X1,Y1)
DF_Thita10cost2, DF_Error10cost2 = DataFrame_Linear_Regression2(TrainX1, TrainY1, TestX1, TestY1, No_of_iteration)
DF_Thita10cost2.to_csv("Parameters_10_cost2.csv", sep='\t', encoding='utf-8')
DF_Error10cost2.to_csv("Train_Test_Error_10_cost2.csv", sep='\t', encoding='utf-8')


# =================================== PART4.b======================================== #

def gradient_descent_Cost1(X, Y, alpha, degree, iteration):
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
def gradient_descent_Cost2(X, Y, alpha, degree, iteration):
    m = len(X)
    # Parameters
    Thita = np.zeros(degree+1)
    for itr in range(0,iteration):
        
        # Hypothesis
        H = np.zeros(m)
        for i in range(0,degree+1):
            H = H+ Thita[i]*(np.power(X,i))
            
        # Cost
        Cost = (1.0/(2.0*m))*( sum( ((H-Y)*(H-Y))*((H-Y)*(H-Y)) ) )
        
        # Gradient Descent
        for i in range(0,degree+1):
            Temp = ((H-Y)*(H-Y)*(H-Y))*(np.power(X,i))
            Thita[i] = Thita[i] - (alpha*(2.0/m)*(sum(Temp)))
    
    return Thita
def gradient_descent_Cost3(X, Y, alpha, degree, iteration):
    m = len(X)
    # Parameters
    Thita = np.zeros(degree+1)
    for itr in range(0,iteration):
        
        # Hypothesis
        H = np.zeros(m)
        for i in range(0,degree+1):
            H = H+ Thita[i]*(np.power(X,i))
            
        # Cost
        Cost = (1.0/(2.0*m))*( sum( abs(H-Y) ) )
        
        # Gradient Descent
        t=0
        for i in range(0,degree+1):
            for k in range(len(H-Y)):
                if((H-Y)[k] > 0):
                    t += (np.power(X[k],i))
                else:
                    t -= (np.power(X[k],i))
            Thita[i] = Thita[i] - (alpha*(1.0/(2.0*m))*(t))
    return Thita

ALPHA = np.array([0.025,0.05,0.1,0.2,0.5])
for degree in range(9,0,-1):
    RMSE_Y1 =[]
    RMSE_Y2 =[]
    RMSE_Y3 =[]
    print 'plotting for N='+str(degree)
    for alpha in ALPHA:
        Thita = gradient_descent_Cost1(X1, Y1, alpha, degree,No_of_iteration)
        H     = Predict(X1, Thita)
        RMSE  = np.sqrt(sum((H-Y1)*(H-Y1)))
        RMSE_Y1.append(RMSE)

        Thita = gradient_descent_Cost2(X1, Y1, alpha, degree,No_of_iteration)
        H     = Predict(X1, Thita)
        RMSE  = np.sqrt(sum((H-Y1)*(H-Y1)))
        RMSE_Y2.append(RMSE)
    
        Thita = gradient_descent_Cost3(X1, Y1, alpha, degree,No_of_iteration)
        H     = Predict(X1, Thita)
        RMSE  = np.sqrt(sum((H-Y1)*(H-Y1)))
        RMSE_Y3.append(RMSE)
        
        
    plt.plot(ALPHA, RMSE_Y1, color='g', marker='o', linestyle='--', label='Squared Error')
    plt.plot(ALPHA, RMSE_Y2, color='b', marker='o', linestyle='--', label='Fourth power Error')
    plt.plot(ALPHA, RMSE_Y3, color='r', marker='o', linestyle='--', label='Mean absolute Error')
    plt.xlabel('Learning Rate Alpha')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Learning Rate when N='+str(degree))
    plt.legend()
    plt.savefig('RMSE_N'+str(degree)+'.png')
    # plt.show()
    plt.close()