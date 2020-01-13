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
################ Part 1 Implementation #################
########################################################

# Module for Preprocess the data
def Preprocess(path1, path2, split_ratio, CountTokens):
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
def DataLoader(Train, Train_Label, batch_size):
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
def WeightInitialiser(DL1, DL2, DL3):    
    W12 = np.random.uniform(-1,1,size=(DL2,DL1)) 
    W23 = np.random.uniform(-1,1,size=(DL3,DL2))
    B12 = np.random.uniform(-1,1, size=(DL2, 1))
    B23 = np.random.uniform(-1,1, size=(DL3, 1))
    return W12, W23, B12, B23

# Module for performing ForwardPropagation
def forward(data, data_label, W12, W23, B12, B23):
    batch = data.values
    Y     = data_label.values
    m = batch.shape[1]

    assert(batch.shape[1]==Y.shape[1])

    S1 = batch
    X1 = F1(S1)

    S2 = np.dot(W12,X1) + B12
    X2 = Relu(S2)

    S3 = np.dot(W23,X2) + B23
    X3 = Sigmoid(S3)
    return S1, X1, S2, X2, S3, X3, m

# Module for performing BackwardPropagation
def backward(Y, S1, X1, S2, X2, S3, X3, m, W12, W23, B12, B23, alpha):
    Y = Y.values
    
    delta3 = X3-Y
    dW23 = np.dot(delta3,X2.T)/m

    delta2 = np.dot(W23.T,delta3)*Relu_der(S2)
    dW12 = np.dot(delta2, X1.T)/m

    db23 = (np.sum(delta3, axis=1, keepdims=True))/m
    db12 = (np.sum(delta2, axis=1, keepdims=True))/m

    W12 = W12 - alpha*dW12
    W23 = W23 - alpha*dW23
    B12 = B12 - alpha*db12
    B23 = B23 - alpha*db23
    return W12, W23, B12, B23

# Module to train the model and get train and test accuracies and errors at various epochs
def training(Train, Train_Label, DL1, DL2, DL3, alpha, epoch, thresh, Batch_size, Test=None, Test_Label=None):
    W12, W23, B12, B23 = WeightInitialiser(DL1, DL2, DL3)
    Batch = DataLoader(Train, Train_Label, Batch_size)
    report_at = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
    for ITR in range(1,epoch+1):
        for k in Batch:
            data = Batch[k]
            data_train = data[:data.shape[0]-1]
            data_label = data[data.shape[0]-1:]
            S1, X1, S2, X2, S3, X3, m = forward(data_train, data_label, W12, W23, B12, B23)
            W12, W23, B12, B23 = backward(data_label, S1, X1, S2, X2, S3, X3, m, W12, W23, B12, B23, alpha)
        if(ITR%100==0 or ITR in report_at):
            acc_train = test(Train, Train_Label, W12, W23, B12, B23, thresh)
            s1, x1, s2, x2, s3, x3, m_ = forward(Train, Train_Label, W12, W23, B12, B23)
            e  = round(Error(x3, Train_Label.values),5)
            s1, x1, s2, x2, s3, x3, m_ = forward(Test,  Test_Label,  W12, W23, B12, B23)
            e2 = round(Error(x3, Test_Label.values),5)
            if(Test is not None and Test_Label is not None):
                acc_test = test(Test, Test_Label, W12, W23, B12, B23, thresh)
                print("Epoch {}=> Train Accu:{}  Test Accu:{}  Train Error:{}  Test Error:{}".format(ITR,round(acc_train*100,5),round(acc_test*100,5),e,e2))
            else:
                print("Epoch {}=> Train Accu:{}  Train Error:{}".format(ITR,round(acc_train*100,5),e))
    return W12, W23, B12, B23

# Module to test the performance
def test(Test, Test_Label, W12, W23, B12, B23, thresh):
    S1, X1, S2, X2, S3, X3, m = forward(Test, Test_Label, W12, W23, B12, B23)
    counter = 0
    for i in X3[0]:
        if i > thresh:
            X3[0][counter]=1
        else:
            X3[0][counter]=0
        counter+=1
    correct = np.sum(np.sum(np.logical_not(np.logical_xor(X3,Test_Label))))
    accuracy = correct/Test_Label.shape[1]
    return accuracy

####################################################################
########## Part1: Finally Training and Testing the Model ###########
####################################################################

# PreProcessing
split_ratio = 0.8 # 80% training 20% test
Freq_limit  = 500 # top 500 in frequencies
Train, Test, Train_Label, Test_Label = Preprocess('Assignment_4_data.txt', 'stop_words.txt', split_ratio , Freq_limit)

# Training
d1 = 500             # Number of neurons in input layer
d2 = 100             # Number of neurons in hidden layer
d3 = 1               # Number of neurons in output layer
alpha = 0.1          # Learning Rate
Num_Epochs = 2000    # Number of Epochs
Threshold = 0.5      # Threshold for classifying HAM or SPAM
Batch_size  = 50     # Batch Size for creating batches in DataLoader Module

W12, W23, B12, B23 = training(Train, Train_Label, d1, d2, d3, alpha, Num_Epochs, Threshold, Batch_size, Test, Test_Label)

# Testing
Test_Acc = test(Test, Test_Label, W12, W23, B12, B23, Threshold)
print("Test Accuracy: {}".format(Test_Acc*100))