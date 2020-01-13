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
def give_space(count):
    for i in range(count):
        print("\t",end="")
            
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


def GINI(data):
    headers = data.columns.values
    L = data[headers[-1]].value_counts()
    gini = 1
    for i in range(len(L)):
        gini-=(L.iloc[i]/data.shape[0])**2
    return gini

def ENTROPY(data):
    headers = data.columns.values
    L = data[headers[-1]].value_counts()
    entropy = 0
    for i in range(len(L)):
        x = (L.iloc[i]/data.shape[0])
        entropy-=(x)*log(x,2)
    return entropy

def Get_Column_GINI(data):
    if(len(data) ==0):
        return "",[]
    G = GINI(data)
    headers = data.columns.values
    answer_g = 1000000
    answer_col =""
    for col in headers:
        if(headers[-1] != col):
            Rows_count=data[col].value_counts()
            temp = 0
            for value in Rows_count.index.values:
                query = str(col)+str('=="')+str(value)+str('"')
                try:
                    D = (data.query(query)).drop([str(col)], axis=1)
                except:
                    D = Get_Table_with_only(data,col,value)
                g = GINI(D)
                temp += (1.0*(g*D.shape[0])/data.shape[0])
            if ( temp<G and temp <= answer_g):
                answer_col = col
                answer_g = temp
    if(answer_col != ""):
        diff_values = list(data[answer_col].value_counts().index)
        return answer_col,diff_values
    else:
        return answer_col , []


def Get_Column_ENTROPY(data):
    if(len(data) ==0):
        return "",[]
    G = ENTROPY(data)
    headers = data.columns.values
    answer_g = 1000000
    answer_col =""
    for col in headers:
        if(headers[-1] != col):
            Rows_count=data[col].value_counts()
            temp = 0
            for value in Rows_count.index.values:    
                
                query = str(col)+str('=="')+str(value)+str('"')
                try:
                    D = data.query(query)
                except:
                    D = Get_Table_with_only(data,col,value)                
                g = ENTROPY(D)
                temp += (1.0*(g*D.shape[0])/data.shape[0])
            if ( temp<G and temp <= answer_g):
                answer_col = col
                answer_g = temp
    if(answer_col != ""):
        diff_values = list(data[answer_col].value_counts().index)
        return answer_col,diff_values
    else:
        return answer_col , []


def Get_Info_gain_root_ENTROPY(data):
    G = ENTROPY(data)
    print('Entropy of Root:',G)
    headers = data.columns.values
    answer_g = 1000000
    answer_col =""
    for col in headers:
        if(headers[-1] != col):
            Rows_count=data[col].value_counts()
            temp = 0
            for value in Rows_count.index.values:    
                query = str(col)+str('=="')+str(value)+str('"')
                try:
                    D = data.query(query)
                except:
                    D = Get_Table_with_only(data,col,value)                
                g = ENTROPY(D)
                temp += (1.0*(g*D.shape[0])/data.shape[0])
            if ( temp<G and temp <= answer_g):
                answer_col = col
                answer_g = temp
    return G-answer_g

def Get_root_GINI_index(data):
    G = GINI(data)
    headers = data.columns.values
    answer_g = 1000000
    answer_col =""
    for col in headers:
        if(headers[-1] != col):
            Rows_count=data[col].value_counts()
            temp = 0
            for value in Rows_count.index.values:
                query = str(col)+str('=="')+str(value)+str('"')
                try:
                    D = (data.query(query)).drop([str(col)], axis=1)
                except:
                    D = Get_Table_with_only(data,col,value)
                g = GINI(D)
                temp += (1.0*(g*D.shape[0])/data.shape[0])
            if ( temp<G and temp <= answer_g):
                answer_col = col
                answer_g = temp
    return G, answer_g

def get_tree_GINI(data, height=None):
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
    name, values = Get_Column_GINI(data)
    if(len(values)==0):
        answer, probability = prob(data)
        L = Leaf(answer, probability)
        return L
    N = Node(name,values)
    for value in values:
        New_data = Get_Table_with_only(data, name,value) 
        if(height==None):
            New_node = get_tree_GINI(New_data)        
        else:
            New_node = get_tree_GINI(New_data, height-1)
        N.next[value] = New_node
    return N

def get_tree_ENTROPY(data, height=None):
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
    
    name, values = Get_Column_ENTROPY(data)
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
            New_node = get_tree_ENTROPY(New_data)        
        else:
            New_node = get_tree_ENTROPY(New_data, height-1)
        N.next[value] = New_node
    return N



# Training decision tree classifier on the train-data
data         = pd.read_csv('dataset for part 1 - Training Data.csv')
test_data    = pd.read_csv('dataset for part 1 - Test Data.csv')
tree_gini    = get_tree_GINI(data)
tree_entropy = get_tree_ENTROPY(data)

# Printing out the decision tree (Gini Index)
print("****************************************************************")
print("Printing out the decision tree (Gini Index)")
print_tree(tree_gini)
print("****************************************************************\n")


# Printing out the decision tree (Information Gain)
print("****************************************************************")
print("Printing out the decision tree (Information Gain)")
print_tree(tree_entropy)
print("****************************************************************\n")


# Value of Information Gain and Gini Index of the root node
print("****************************************************************")
print("Value of Information Gain and Gini Index of the root node")
root_g, split = Get_root_GINI_index(data)
print("Gini index of Root:\t", root_g, "\nGini(split) at Root:\t", split,"\n")

root_e = Get_Info_gain_root_ENTROPY(data)
print("Information Gain of root:", root_e)
print("****************************************************************\n")


# Prediction on Train-data (Gini Index)
print("****************************************************************")
print("Prediction on Train-data (Gini Index)")
print(predict(tree_gini, data))
print("****************************************************************\n")


# Prediction on Test-data (Gini Index)
print("****************************************************************")
print("Prediction on Test-data (Gini Index)")
print(predict(tree_gini, test_data))
print("****************************************************************\n")


# Prediction on Train-data (Information Gain)
print("****************************************************************")
print("Prediction on Train-data (Information Gain)")
print(predict(tree_entropy, data))
print("****************************************************************\n")


# Prediction on Test-data (Information Gain)
print("****************************************************************")
print("Prediction on Test-data (Information Gain)")
print(predict(tree_entropy, test_data))
print("****************************************************************\n")


# Converting Dataset to 0 and 1
data = pd.read_csv('dataset for part 1 - Training Data.csv')
testdata = pd.read_csv('dataset for part 1 - Test Data.csv')

data[data.columns[0]]  = data[data.columns[0]].astype(str)
data[data.columns[1]]  = data[data.columns[1]].astype(str)
data[data.columns[2]]  = data[data.columns[2]].astype(str)
data[data.columns[3]]  = data[data.columns[3]].map(dict(yes=1, no=0))
data[data.columns[4]]  = data[data.columns[4]].map(dict(yes=1, no=0))

testdata[testdata.columns[0]]  = testdata[testdata.columns[0]].astype(str)
testdata[testdata.columns[1]]  = testdata[testdata.columns[1]].astype(str)
testdata[testdata.columns[2]]  = testdata[testdata.columns[2]].astype(str)
testdata[testdata.columns[3]]  = testdata[testdata.columns[3]].map(dict(yes=1, no=0))
testdata[testdata.columns[4]]  = testdata[testdata.columns[4]].map(dict(yes=1, no=0))

Train = pd.get_dummies(data, prefix=['price','maintenance','capacity'])
Train = Train[['airbag', 'price_high', 'price_low', 'price_med', 'maintenance_high', 'maintenance_low', 'maintenance_med', 'capacity_2','capacity_4', 'capacity_5', 'profitable']]
Train = Train.applymap(int)

Test = pd.get_dummies(testdata, prefix=['price','maintenance','capacity'])
for col in Train.columns:
    try:
        Test[col]
    except:
        l = len(Test)
        Test[col] = [0]*l

Test = Test[['airbag', 'price_high', 'price_low', 'price_med', 'maintenance_high', 'maintenance_low', 'maintenance_med', 'capacity_2','capacity_4', 'capacity_5', 'profitable']]
Test = Test.applymap(int)


# Printing out the decision tree (Gini Index) (dummies used)
print("****************************************************************")
print("Printing out the decision tree (Gini Index) (dummies used)")
tree_gini_dummy = get_tree_GINI(Train)
print_tree(tree_gini_dummy)
print("****************************************************************\n")


# Printing out the decision tree (Information Gain) (dummies used)
print("****************************************************************")
print("Printing out the decision tree (Information Gain) (dummies used)")
tree_entropy_dummy    = get_tree_ENTROPY(Train)
print_tree(tree_entropy_dummy)
print("****************************************************************\n")


# Value of Information Gain and Gini Index of the root node (Dummies used)
print("****************************************************************")
print("Value of Information Gain and Gini Index of the root node (Dummies used)")
root_g, split = Get_root_GINI_index(data)
print("Gini index of Root:\t", root_g, "\nGini(split) at Root:\t", split,"\n")

root_e = Get_Info_gain_root_ENTROPY(data)
print("Information Gain of root:", root_e)
print("****************************************************************\n")


# Prediction on Train-data (Gini Index) (dummies used)
print("****************************************************************")
print("Prediction on Train-data (Gini Index) (dummies used)")
print(predict(tree_gini_dummy, Train))
print("****************************************************************\n")


# Prediction on Test-data (Gini Index) (dummies used)
print("****************************************************************")
print("Prediction on Test-data (Gini Index) (dummies used)")
print(predict(tree_gini_dummy, Test))
print("****************************************************************\n")


# Prediction on Train-data (Information Gain) (dummies used)
print("****************************************************************")
print("Prediction on Train-data (Information Gain) (dummies used)")
print(predict(tree_entropy_dummy, Train))
print("****************************************************************\n")


# Prediction on Test-data (Information Gain) (dummies used)
print("****************************************************************")
print("Prediction on Test-data (Information Gain) (dummies used)")
print(predict(tree_entropy_dummy, Test))
print("****************************************************************\n")


## Scikit-Learn Decision Tree
# Training Models by Gini Index and Information Gain
# By Scikit-Learn Decision Tree
data    = pd.read_csv('dataset for part 1 - Training Data.csv')
Y_ML = data['profitable']
Y_ML = Y_ML.map(dict(yes=1, no=0))
X_ML = data.drop('profitable', axis=1)  

X_ML[X_ML.columns[0]]  = X_ML[X_ML.columns[0]].astype(str)
X_ML[X_ML.columns[1]]  = X_ML[X_ML.columns[1]].astype(str)
X_ML[X_ML.columns[2]]  = X_ML[X_ML.columns[2]].astype(str)
X_ML[X_ML.columns[3]]  = X_ML[X_ML.columns[3]].map(dict(yes=1, no=0))

Train_X = pd.get_dummies(X_ML, prefix=['price','maintenance','capacity'])
Train_X = Train_X.applymap(int)

MODEL = DecisionTreeClassifier(criterion='gini')  
MODEL = MODEL.fit(Train_X, Y_ML)  

Testdata= pd.read_csv('dataset for part 1 - Test Data.csv')
Test_Y  = Testdata['profitable']
Test_Y  = Test_Y.map(dict(yes=1, no=0))

Test_X  = Testdata.drop('profitable', axis=1)

Test_X[Test_X.columns[0]]  = Test_X[Test_X.columns[0]].astype(str)
Test_X[Test_X.columns[1]]  = Test_X[Test_X.columns[1]].astype(str)
Test_X[Test_X.columns[2]]  = Test_X[Test_X.columns[2]].astype(str)
Test_X[Test_X.columns[3]]  = Test_X[Test_X.columns[3]].map(dict(yes=1, no=0))

TestX_ML = pd.get_dummies(Test_X, prefix=['price','maintenance','capacity'])
for col in Train_X.columns:
    try:
        TestX_ML[col]
    except:
        l = len(TestX_ML)
        TestX_ML[col] = [0]*l
TestX_ML = TestX_ML[['airbag', 'price_high', 'price_low', 'price_med', 'maintenance_high', 'maintenance_low', 'maintenance_med', 'capacity_2','capacity_4', 'capacity_5']]
TestX_ML = TestX_ML.applymap(int)

MODEL2 = DecisionTreeClassifier(criterion="entropy")  
MODEL2 = MODEL2.fit(Train_X, Y_ML)  

# Prediction by Scikit Learn model on Train-Data
print("****************************************************************")
print("Prediction by Scikit Learn model on Train-Data")
print("Training Labels:")
print('\t\t', np.array(Y_ML))
print("\n*********************Prediction on TRAIN DATA***********************")
print("\nModel Prediction on Train Data (Scikit-Learn) (Gini Index) (dummy)\n")
print("\t\t", MODEL.predict(Train_X))
print("\nModel Prediction on Train Data (Scikit-Learn) (Information Gain) (dummy)\n")
print("\t\t", MODEL2.predict(Train_X))
print("****************************************************************\n")


# Prediction by Scikit Learn model on Test-Data
print("****************************************************************")
print("Prediction by Scikit Learn model on Test-Data")
print("Testing Labels:")
print('\t\t', np.array(Test_Y))
print("\n*********************Prediction on TEST DATA***********************")
print("\nModel Prediction on Test Data (Scikit-Learn) (Gini Index) (dummy)\n")
print("\t\t", MODEL.predict(TestX_ML))
print("\nModel Prediction on Test Data (Scikit-Learn) (Information Gain) (dummy)\n")
print("\t\t", MODEL2.predict(TestX_ML))
print("****************************************************************\n")


# Visualization of the Decision Tree created by Scikit Learn (Gini Index)
dotfile  = open("tree_ML_gini.dot",'w')
dot_data = tree.export_graphviz(MODEL, out_file=dotfile, filled = True, rounded= True, special_characters=True, feature_names=Train_X.columns)
dotfile.close()
os.system("dot -Tpng tree_ML_gini.dot > output_gini.png")
img      = mpimg.imread('output_gini.png')
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 16), dpi=80, facecolor='w', edgecolor='k')
imgplot  = plt.imshow(img)
os.system("rm tree_ML_gini.dot")
plt.show()
plt.close()

# Visualization of the Decision Tree created by Scikit Learn (Information Gain)
dotfile  = open("tree_ML_entropy.dot",'w')
dot_data = tree.export_graphviz(MODEL2, out_file=dotfile, filled = True, rounded= True, special_characters=True, feature_names=Train_X.columns)
dotfile.close()
os.system("dot -Tpng tree_ML_entropy.dot > output_entropy.png")
img      = mpimg.imread('output_entropy.png')
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 16), dpi=80, facecolor='w', edgecolor='k')
imgplot  = plt.imshow(img)
os.system("rm tree_ML_entropy.dot")
plt.show()
plt.close()