import pandas as pd
import numpy as np
from math import log
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
class Leaf:
    def __init__(self,ans,probability):
        self.next = None
        self.ans  = ans
        self.probability = probability
        
class Node:
    def __init__(self,question,values):
        d = dict()
        for value in values:
            d[value]=None
        self.next = d
        self.question = question

def Get_Table_with_only(data , col, value):
    to_remove = []
    for index, row in data.iterrows():
        if(row[col] != value):
            to_remove.append(index)
    Y = data.drop(to_remove)
    Y.drop(col, axis=1, inplace=True)
    Y = Y.reset_index(drop=True)
    return Y

def get_Id_to_word(words):
    Id_to_Word = dict()
    for Id in range(1,len(words)+1):
        Id_to_Word[Id]=((words.iat[Id-1,0]).strip())
    return Id_to_Word

def get_docs_by_ids(traindata):
    docs = dict()
    for i in range(len(traindata)):
        ID = traindata.iat[i,0]
        V  = traindata.iat[i,1]
        if(ID in docs):
            docs[ID].add(V);
        else:
            docs[ID]=set()
            docs[ID].add(V)
    return docs
def get_docid_to_label(trainlabel):
    labels = dict()
    for Id in range(1,len(trainlabel)+1):
        labels[Id]=int(trainlabel.iat[Id-1,0])
    return labels

def create_dataset(WID_W, DID_WID, DID_L):
    col_len = len(WID_W)
    cols = []
    for w in WID_W:
        cols.append(WID_W[w])
    assert(len(cols)==col_len)
    Data = list()
    for DID in DID_WID:
        new_row = markone(DID_WID[DID],col_len)
        new_row.append(DID_L[DID]-1)
        Data.append(new_row)
    
    
    cols.append("Output")
    Data = pd.DataFrame(Data, columns=cols)
    return Data

def markone(S,col_len):
    A = [0]*col_len
    for i in S:
        A[i-1]=1
    return A

def GET_DATASET(path1, path2, wordid_to_word):
    traindata       = pd.read_csv(path1, sep='\t', header=None, keep_default_na=False)
    docid_to_wordid = get_docs_by_ids(traindata)

    trainlabel      = pd.read_csv(path2, sep='\t', header=None)
    docid_to_label  = get_docid_to_label(trainlabel)

    Dataset         = create_dataset(wordid_to_word, docid_to_wordid, docid_to_label)
    return Dataset

def Predict_on_SL_model(Dataset, model):
    ANS = model.predict(X)
    count = 0
    for i in range(len(Dataset)):
        if(Dataset.iat[i,-1]!=ANS[i]):
            count+=1
    print ("Number of misclassification:", count, "out of ", len(Dataset))
    return count,len(Dataset)

def accuracy(Real, Pred):
    assert(len(Real)==len(Pred))
    correct = 0
    incorrect = 0
    for i in range(len(Real)):
        if((Real[i]==0 and Pred[i]=='no')or(Real[i]==1 and Pred[i]=='yes')):
            correct+=1
        else:
            incorrect+=1
    assert((correct+incorrect) == len(Real))
    return (correct/len(Real))*100

def prediction_partB(TREE, Dataset, Test_Dataset):
    print("Predicting on Train Data (Information Gain)..")
    TrainPred  = predict(TREE, Dataset)
    TrainAccu  = accuracy(TrainPred['Output'],TrainPred['Predicted'])
    print('Training Accuracy :',TrainAccu,'%')

    print("Predicting on Test Data (Information Gain)..")
    TestPred   = predict(TREE, Test_Dataset)
    TestAccu   = accuracy(TestPred['Output'],TestPred['Predicted'])
    print('Training Accuracy :',TestAccu,'%')
    return [TrainAccu, TestAccu]

def ENTROPY(data):
    headers = data.columns.values
    L = data[headers[-1]].value_counts()
    entropy = 0
    for i in range(len(L)):
        x = (L.iloc[i]/data.shape[0])
        entropy-=(x)*log(x,2)
    return entropy

def GET_TREE_ENTROPY(data, height=None):
    if(height==None):
        if(len(data.columns.values)==1):
            answer, probability = prob(data)
            L = Leaf(answer, probability)
            return L
    else:
        if((len(data.columns.values)==1) or (height<=0)):
            answer, probability = prob(data)
            L = Leaf(answer, probability)
            return L        
        
    name, values = GET_COL_ENTROPY(data)
    if(len(name)>0 and height is not None):
        print("HEIGHT:", height, "NAME:",name)
    if(len(values)==0):
        answer, probability = prob(data)
        L = Leaf(answer, probability)
        return L
    N = Node(name,values)
    
    for value in values:
        New_data = Get_Table_with_only(data, name,value)
        if(height==None):
            New_node = GET_TREE_ENTROPY(New_data)        
        else:
            New_node = GET_TREE_ENTROPY(New_data, height-1)
        N.next[value] = New_node
    return N

def GET_COL_ENTROPY(data):
    if(len(data) ==0):
        return "",[]
    G = ENTROPY(data)
    headers = data.columns.values

    answer_g = 1000000
    answer_col =""
    answer_col = get_Best_node(data)
    if(answer_col != ""):
        diff_values = list(data[answer_col].value_counts().index)
        return answer_col,diff_values
    else:
        return answer_col , []


def get_Best_node(Dataset):
    D = Dataset.values
    p1 = np.sum(D[:,-1])/D.shape[0]
    p2 = 1-p1
    E = 0
    if(p1!=0):
        E  -= p1*log(p1,2)
    if(p2!=0):
        E  -= p2*log(p2,2)
    ans = 1000000
    ans_col =""
    columns = Dataset.columns.values
    for col in range(len(columns)-1):
        A = D[:,col]
        B = D[:,-1]
        S = A.shape[0]
        A = A.reshape((S,1))
        B = B.reshape((S,1))
        NA = np.logical_not(A)
        NB = np.logical_not(B)
        OO = np.bitwise_and(A,B)
        OZ = np.bitwise_and(A, NB)
        ZO = np.bitwise_and(NA, B)
        ZZ = np.bitwise_and(NA, NB)
        OO = np.sum(OO)
        ZO = np.sum(ZO)
        OZ = np.sum(OZ)
        ZZ = np.sum(ZZ)
        assert(OO+OZ+ZO+ZZ == S)
        N1 = np.sum(A).reshape((1,1))
        NO = len(A)-N1
        tm  =0
        tm2 =0
        try:
            tm -= (OO/(OO+OZ))*log((OO/(OO+OZ)),2)
        except:
            pass
        try:
            tm -= (OZ/(OO+OZ))*log((OZ/(OO+OZ)),2) 
        except:
            pass
        try:
            tm2 -= (ZO/(ZO+ZZ))*log((ZO/(ZO+ZZ)),2)
        except:
            pass
        try:
            tm2 -=  (ZZ/(ZO+ZZ))*log((ZZ/(ZO+ZZ)),2)
        except:
            pass

        WE1 = (N1[0][0]/len(A))*tm
        WEO = (NO[0][0]/len(A))*tm2
        T  = (WEO+WE1)
        if( T <=ans and T < E):
            ans = T
            ans_col = columns[col]
    return ans_col

def prob(data):
    headers = data.columns.values
    Rows_count=data[headers[-1]].value_counts()        
    yes = 0
    no  = 0
    try:
        yes = Rows_count['yes']
    except:
        try:
            yes = Rows_count[1]
        except:
            pass
    try:
        no  = Rows_count['no']
    except:
        try:
            no = Rows_count[0]
        except:
            pass
    if(yes>=no):
        return "yes", yes/(yes+no)
    else:
        return "no", no/(yes+no)


def print_tree(Tree,count=0):
    if(Tree==None):
        print("\n",end="")
        return
    elif(Tree.next==None):
        print(" : ",end="")
        print(Tree.ans)
        return
    else:
        print("\n",end="")        
        for k in Tree.next:
            for i in range(count):
                print("\t",end="")
            print("| "+Tree.question+" = "+str(k),end="")
            print_tree(Tree.next[k],count+1)

def predict(tree_gini,test_data):
    L = []
    header = list(test_data.columns)
    header.append('Predicted')
    for i in range(len(test_data)):
        l=list(test_data.loc[i])
        l.append(predict_util(tree_gini, test_data, i))
        L.append(l)
    df = pd.DataFrame(L,columns=header)
    return df

def predict_util(Tree,data,row):
    if(Tree==None):
        return 'dont_know'
    elif(Tree.next==None):
        return Tree.ans
    else:
        return predict_util(Tree.next[data[Tree.question][row]], data, row)



# Creating Dataset
words           = pd.read_csv('words.txt', header=None, keep_default_na=False) #prevent from reading null as NaN
wordid_to_word  = get_Id_to_word(words)

Dataset         = GET_DATASET('traindata.txt','trainlabel.txt',wordid_to_word)
Test_Dataset    = GET_DATASET('testdata.txt','testlabel.txt',wordid_to_word)


# By Scikit Learn
Y = Dataset['Output']
X = Dataset.drop('Output', axis=1)  
MODEL3 = DecisionTreeClassifier(criterion='entropy')  
MODEL3 = MODEL3.fit(X, Y)

print("On Train-Data:")
x,y = Predict_on_SL_model(Dataset, MODEL3)
print("Accuracy:",(y-x)*100/y,'%\n')
print("On Test-Data:")
x,y = Predict_on_SL_model(Test_Dataset, MODEL3)
print("Accuracy:",(y-x)*100/y,'%\n')


# Visualization of the Decision Tree created by Scikit Learn (Information Gain)
dotfile  = open("tree_ML_entropy_3.dot",'w')
dot_data = tree.export_graphviz(MODEL3, out_file=dotfile, filled = True, rounded= True, special_characters=True, feature_names=Dataset.columns[:-1])
dotfile.close()
os.system("dot -Tpng tree_ML_entropy_3.dot > output_entropy_3.png")
img      = mpimg.imread('output_entropy_3.png')
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 16), dpi=80, facecolor='w', edgecolor='k')
imgplot  = plt.imshow(img)
os.system("rm tree_ML_entropy_3.dot")
plt.show()
plt.close()


## By My Model
Results = []
Trees = dict()
for height in range(1,25):
    print("*****************************************************************")
    print('When Maximum Height:',height)
    print('Training Decision Tree...')
    TREE = GET_TREE_ENTROPY(Dataset,height)
    Trees[height] = TREE
    print("\n")
    Res  = prediction_partB(TREE, Dataset, Test_Dataset)
    Res.append(height)
    print(Res)
    Results.append(Res)
    print("*****************************************************************\n\n")    


# Train Accuracy and Test Accuracy VS Maximum Tree Depth
print("*****************************************************************")
print("Train Accuracy and Test Accuracy VS Maximum Tree Depth")
TR_Res = []
TE_Res = []
Depth  = []
counter = 1
for r in Results:
    TR_Res.append(r[0])
    TE_Res.append(r[1])
    Depth.append(r[2])
    assert(counter==r[2])
    counter+=1
Acc_Df = pd.DataFrame()
Acc_Df['Max_Depth']=pd.Series(Depth)
Acc_Df['Train_Accuracy']=pd.Series(TR_Res)
Acc_Df['Test_Accuracy']=pd.Series(TE_Res)
print(Acc_Df)
print("*****************************************************************\n\n")



# Plot of Train Accuracy and Test Accuracy VS Maximum Tree Depth
TR_Res = np.array(TR_Res)
TE_Res = np.array(TE_Res)
Depth  = np.array(Depth)

fig = plt.figure(figsize=(12, 6))
plt.plot(Depth, TR_Res, color='b', linestyle='-', label='Train Accuracy')
plt.plot(Depth, TE_Res, color='r', linestyle='-', label='Test Accuracy')
plt.axvline(x=21, color='g', linestyle='--',label='Full growth of tree')

plt.xlabel('Max Depth of the Tree')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('PartB_Accuracy.png')
plt.grid()
plt.show()
plt.close()


# Printing Full Grown Tree
print("*****************************************************************")
print("Printing Full Grown Tree")
print_tree(Trees[24])
print("*****************************************************************\n\n")


# Maximum Accuracy Tree
print("*****************************************************************")
print("Maximum Accuracy Tree")
print_tree(Trees[9])
print("*****************************************************************\n\n")