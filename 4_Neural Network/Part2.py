import pandas as pd
import numpy as np
import sys
import numpy
import math
import networkx as nx
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=sys.maxsize)
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.special import xlogy
import operator

########################################################
######## Modules required by Preprocess Module #########
########################################################

def get_stop_words(path):
    data = pd.read_csv(path, sep='\n', header=None)
    temp = []
    for i in range(len(data)):
        x = str(data.iloc[i,0])
        temp.append(x.lower())
    stop_words = temp
    return stop_words

def get_train_test_data(data, label, train_percent):
    data = pd.DataFrame(data.T)
    label = pd.DataFrame(label.T)
    m = np.random.rand(len(data)) < train_percent
    train = data[m]
    train_label = label[m]
    test  = data[~m]
    test_label = label[~m]
    return pd.DataFrame(train.T), pd.DataFrame(test.T), pd.DataFrame(train_label.T), pd.DataFrame(test_label.T)

def get_top_tokens(Freq, limit):
    res = []
    for i in range(min(limit,len(Freq))):
        res.append(Freq[i][0])
    return res

def New_get_All_Tokens(data, stop_words):
    L=[]
    ps = PorterStemmer()
    for row in range(data.shape[0]):
        sen = data.iloc[row,1]
        sen = sen.lower()
        x = New_get_list(sen,stop_words)
        L = L+x
    X = []
    for word in L:
        if word in stop_words:
            pass
        elif ps.stem(word) in stop_words:
            pass
        else:
            X.append(ps.stem(word))
    L = X
    D = dict()
    for word in L:
        if word in D:
            D[word]+=1
        else:
            D[word]=1
    D = sorted(D.items(), key=operator.itemgetter(1))
    D.reverse()
    return set(L), D

def New_get_list(s,stop_words):
    ps = PorterStemmer()
    start = 0
    counter=-1
    L=[]
    s = s.lower()
    for i in range(len(s)):
        counter+=1
        i = s[i]
        if(i==' ' or i=='\t' or i=='\n' or i=='.' or i==',' or i==':' or i=='-'):
            temp = s[start:counter]
            if(len(temp)>0):
                L.append(str(temp))
            start=counter+1
        else:
            pass
    if(start <len(s) and len(s[start:])>0):
        temp = s[start:]
        L.append(temp)
    return L

def New_Get_Input_representation(data, Top_Tokens, stop_words):
    DATA  = np.ones((len(Top_Tokens), len(data)))
    LABEL = np.ones((1, len(data)))
    counter = -1
    for row in range(0,data.shape[0]):
        counter += 1
        X = get_token_of_string(data.iloc[row,1], stop_words)
        for i,word in enumerate(Top_Tokens):
            if word in X:
                DATA[i,counter]=1
            else:
                DATA[i,counter]=0
        if(str(data.iloc[row,0])=="spam"):   
            LABEL[0,counter] = 1
        elif(str(data.iloc[row,0])=="ham"):
            LABEL[0,counter] = 0
        else:
            print("ERROR in label at row:",row)
    return DATA, LABEL

def get_token_of_string(sen, stop_words):
    ps = PorterStemmer()
    sen = sen.lower()
    x = New_get_list(sen,stop_words)
    L = x
    X = []
    for word in L:
        if word in stop_words:
            pass
        elif ps.stem(word) in stop_words:
            pass
        else:
            X.append(ps.stem(word))
    L = X
    return set(L)

########################################################
########## Some Activation and Loss Functions ##########
########################################################

# Identity Function
def F1(x):  
    return x

# Relu Activation Function
def Relu(x):
    return np.maximum(0,x)

# Relu Derivative Function
def Relu_der(x):
    x = x>0
    x = x +0
    return x

# Sigmoid Activation Function
def Sigmoid(x):
    s = 1.0/(1.0+np.exp(-x))
    return s

# Sigmoid Derivative Function
def Sigmoid_der(x):
    return Sigmoid(x)*(1-Sigmoid(x))

# Softmax Activation Function
def Softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 0, keepdims = True)
    s = x_exp/x_sum
    return s

# Categorical Cross Entropy Loss
def Error(X, Y):
    E = (xlogy(Y, X) + xlogy(1 - Y, 1 - X))
    E = -(np.sum(E))/(X.shape[1])
    return E


########################################################
################ Part 2 Implementation #################
########################################################

# Module for Preprocess the data
def Preprocess2(path1, path2, split_ratio, CountTokens):
    data = pd.read_csv(path1, sep='\t', header=None)
    stop_words = get_stop_words(path2)
    Tokens, Freq = New_get_All_Tokens(data, stop_words)
    Top_Tokens = get_top_tokens(Freq, 500)
    Dataset, Label = New_Get_Input_representation(data, Top_Tokens, stop_words)
    
    DataSet = pd.DataFrame(Dataset)  # (500, #ex)
    Label_  = pd.DataFrame(Label)    # (1, #ex)
    Train, Test, Train_Label, Test_Label = get_train_test_data(DataSet, Label_,split_ratio)
    Train = pd.DataFrame(Train.values)
    Test  = pd.DataFrame(Test.values)
    Train_Label = pd.DataFrame(Train_Label.values)
    Test_Label = pd.DataFrame(Test_Label.values)    
    return Train, Test, Train_Label, Test_Label

# Module for Loading the data
def DataLoader2(Train, Train_Label, batch_size):
    df1 = Train
    df2 = Train_Label
    Result = pd.concat([df1, df2], axis=0, ignore_index=True)
    mixed = pd.DataFrame(Result.sample(frac=1, axis=1).values)
    size = mixed.shape[1]
    batches = math.ceil(size/batch_size)
    BATCH=dict()
    counter = -1
    for i in range(batches):
        counter+=1
        b = "batch"+str(counter)
        x = (counter)*batch_size
        y = (counter+1)*batch_size
        if((counter+1)*batch_size >= size):
            y = size
        l = [i for i in range(x, y)]
        BATCH[b] = mixed[l]
    return BATCH

# Module for Initialising Weights and Bias
def WeightInitialiser2(DL1, DL2, DL3, DL4):    
    W12 = np.random.uniform(-1,1,size=(DL2,DL1))
    W23 = np.random.uniform(-1,1,size=(DL3,DL2))
    W34 = np.random.uniform(-1,1,size=(DL4,DL3))
    B12 = np.random.uniform(-1,1, size=(DL2, 1))
    B23 = np.random.uniform(-1,1, size=(DL3, 1))
    B34 = np.random.uniform(-1,1, size=(DL4, 1))
    return W12, W23, W34, B12, B23, B34

# Module for performing ForwardPropagation
def forward2(data, data_label, W12, W23, W34, B12, B23, B34):
    batch = data.values
    Y     = data_label.values
    m     = batch.shape[1]

    assert(batch.shape[1]==Y.shape[1])

    S1 = batch
    X1 = S1

    S2 = np.dot(W12,X1) + B12
    X2 = Sigmoid(S2)

    S3 = np.dot(W23,X2) + B23
    X3 = Sigmoid(S3)

    S4 = np.dot(W34,X3) + B34
    X4 = Softmax(S4)

    return S1, X1, S2, X2, S3, X3, S4, X4, m

# Module for performing BackwardPropagation
def backward2(Y, S1, X1, S2, X2, S3, X3, S4, X4, m, W12, W23, W34, B12, B23, B34, alpha):
    Y = Y.values
    Y = np.array([Y[0],1-Y[0]])
     
    delta4 = X4-Y
    dW34 = np.dot(delta4,X3.T)/m

    delta3 = np.dot(W34.T,delta4)*Sigmoid_der(S3)
    dW23 = np.dot(delta3, X2.T)/m

    delta2 = np.dot(W23.T,delta3)*Sigmoid_der(S2)
    dW12 = np.dot(delta2, X1.T)/m

    db34 = (np.sum(delta4, axis=1, keepdims=True))/m
    db23 = (np.sum(delta3, axis=1, keepdims=True))/m
    db12 = (np.sum(delta2, axis=1, keepdims=True))/m

    W12 = W12 - alpha*dW12
    W23 = W23 - alpha*dW23
    W34 = W34 - alpha*dW34
    B12 = B12 - alpha*db12
    B23 = B23 - alpha*db23
    B34 = B34 - alpha*db34
    return W12, W23, W34, B12, B23, B34

# Module to train the model and get train and test accuracies and errors at various epochs
def training2(Train, Train_Label, DL1, DL2, DL3, DL4, alpha, epoch, thresh, Batch_size, Test=None, Test_Label=None):
    W12, W23, W34, B12, B23, B34 = WeightInitialiser2(DL1, DL2, DL3, DL4)
    Batch = DataLoader2(Train, Train_Label, Batch_size)
    report_at = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
    for ITR in range(1,epoch+1):
        for k in Batch:
            data = Batch[k]
            data_train = data[:data.shape[0]-1]
            data_label = data[data.shape[0]-1:]
            S1, X1, S2, X2, S3, X3, S4, X4, m = forward2(data_train, data_label, W12, W23, W34, B12, B23, B34)
            W12, W23, W34, B12, B23, B34 = backward2(data_label, S1, X1, S2, X2, S3, X3, S4, X4, m, W12, W23, W34, B12, B23, B34, alpha)
        if(ITR%100==0 or ITR in report_at):
            acc_train = test2(Train, Train_Label, W12, W23, W34, B12, B23, B34, thresh)

            s1, x1, s2, x2, s3, x3, s4, x4, m = forward2(Train, Train_Label, W12, W23, W34, B12, B23, B34) 
            Y = np.array([Train_Label.values[0],1-Train_Label.values[0]])
            e = -np.sum(xlogy(Y,x4))/x4.shape[1]
            
            s1, x1, s2, x2, s3, x3, s4, x4, m = forward2(Test, Test_Label, W12, W23, W34, B12, B23, B34)
            Y  = np.array([Test_Label.values[0],1-Test_Label.values[0]])
            e2 = -np.sum(xlogy(Y,x4))/x4.shape[1]
            
            if(Test is not None and Test_Label is not None):
                acc_test = test2(Test, Test_Label, W12, W23, W34, B12, B23, B34, thresh)
                print("Epoch {}=> Train Accu:{}  Test Accu:{}  Train Error:{}  Test Error:{}".format(ITR,round(acc_train*100,5),round(acc_test*100,5),round(e,5),round(e2,5)))
            else:
                print("Epoch {}=> Train Accu:{}  Train Error:{}".format(ITR,round(acc_train*100,5),round(e,5)))
    return W12, W23, W34, B12, B23, B34

# Module to test the performance
def test2(Test, Test_Label, W12, W23, W34, B12, B23, B34, thresh):
    S1, X1, S2, X2, S3, X3, S4, X4, m = forward2(Test, Test_Label, W12, W23, W34, B12, B23, B34)
    Y = np.array([Test_Label.values[0],1-Test_Label.values[0]])
    size = X4.shape[1]
    assert(size == Test_Label.shape[1])
    correct = 0
    for i in range(size):
        if X4[0][i] > X4[1][i] and Y[0][i]==1:
            correct+=1
        elif X4[0][i] <= X4[1][i] and Y[1][i]==1:
            correct+=1
        else:
            pass
    accuracy = correct/size
    return accuracy


####################################################################
########## Part2: Finally Training and Testing the Model ###########
####################################################################

# PreProcessing
split_ratio = 0.8 # 80% training 20% test
Freq_limit  = 500 # top 500 in frequencies
Train, Test, Train_Label, Test_Label = Preprocess2('Assignment_4_data.txt', 'stop_words.txt', split_ratio , Freq_limit)

# Training
d1 = 500             # Number of neurons in input layer
d2 = 100             # Number of neurons in hidden layer 1
d3 = 20              # Number of neurons in hidden layer 2
d4 = 2               # Number of neurons in hidden layer 2
alpha = 0.1          # Learning Rate
Num_Epochs = 2000    # Number of Epochs
Threshold = 0.5      # Threshold for classifying HAM or SPAM
Batch_size  = 100    # Batch Size for creating batches in DataLoader Module
W12, W23, W34, B12, B23, B34 = training2(Train, Train_Label, d1, d2, d3, d4, alpha, Num_Epochs, Threshold, Batch_size, Test, Test_Label)

# Testing
Test_Acc = test2(Test, Test_Label, W12, W23, W34, B12, B23, B34, Threshold)
print("Test Accuracy: {}".format(Test_Acc*100))