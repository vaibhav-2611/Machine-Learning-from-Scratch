import pandas as pd
import numpy as np
import sys
import numpy
import math
import networkx as nx
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=sys.maxsize)

def Jaccard_Coeff(S1, S2):
    return len(S1.intersection(S2))/len(S1.union(S2))

def Distance_single_linkage(L1, L2):
    temp = math.inf
    for seti in L1:
        for setj in L2:
            distance = 1-Jaccard_Coeff(seti,setj)
            temp = min(temp,distance)
    return temp

def Distance_complete_linkage(L1, L2):
    temp = -math.inf
    for seti in L1:
        for setj in L2:
            distance = 1-Jaccard_Coeff(seti,setj)
            temp = max(temp,distance)
    return temp

# Hierarchical Clustering method with Single Linkage
def Cluster_Single_Linkage():
    data = pd.read_csv('AAAI.csv')
    Points = dict()
    Docs   = dict()
    for i in range(0,data.shape[0]):
        Points[i]=[]
        Docs[i] = []
        temp = set()
        for item in data.iloc[i,2].split('\n'):
            temp.add(item)
        Points[i].append(temp)
        Docs[i].append(i)

    Number_of_clusters = 9
    Current_number_of_clusters = len(Points)
    while(Current_number_of_clusters>Number_of_clusters):
        temp = math.inf
        for i in Points:
            for j in Points:
                if i==j:
                    continue
                x = Distance_single_linkage(Points[i],Points[j])
                if(temp>x):
                    cluster1 = i
                    cluster2 = j
                    temp = x
        L1 = Points[cluster1]
        L2 = Points[cluster2]
        L = L1 + L2
        Points.pop(cluster1)
        Points.pop(cluster2)
        Points[cluster1] = L
        Current_number_of_clusters -= 1
        L1 = Docs[cluster1]
        L2 = Docs[cluster2]
        L = L1 + L2
        Docs.pop(cluster1)
        Docs.pop(cluster2)
        Docs[cluster1] = L
    for i in Docs:
        Docs[i].sort()
    return Points, Docs

# Hierarchical Clustering method with Complete Linkage
def Cluster_Complete_Linkage():
    data = pd.read_csv('AAAI.csv')
    Points = dict()
    Docs   = dict()
    for i in range(0,data.shape[0]):
        Points[i]=[]
        Docs[i] = []
        temp = set()
        for item in data.iloc[i,2].split('\n'):
            temp.add(item)
        Points[i].append(temp)
        Docs[i].append(i)

    Number_of_clusters = 9
    Current_number_of_clusters = len(Points)
    while(Current_number_of_clusters>Number_of_clusters):
        temp = math.inf
        for i in Points:
            for j in Points:
                if i==j:
                    continue
                x = Distance_complete_linkage(Points[i],Points[j])
                if(temp>x):
                    cluster1 = i
                    cluster2 = j
                    temp = x
        L1 = Points[cluster1]
        L2 = Points[cluster2]
        L = L1 + L2
        Points.pop(cluster1)
        Points.pop(cluster2)
        Points[cluster1] = L
        Current_number_of_clusters -= 1
        L1 = Docs[cluster1]
        L2 = Docs[cluster2]
        L = L1 + L2
        Docs.pop(cluster1)
        Docs.pop(cluster2)
        Docs[cluster1] = L
    for i in Docs:
        Docs[i].sort()
    return Points, Docs

# ------------------------PART1---------------------------
print("PART1:\n")
# Hierarchical Clustering method with Single Linkage
Clusters1, Ids1 = Cluster_Single_Linkage()
counter = 1
print("Clusters By Single Linkage:")
for i in Ids1:
    print("Size:", len(Ids1[i]), "\tCluster",counter,": ",Ids1[i])
    counter +=1

# Hierarchical Clustering method with Complete Linkage
Clusters2, Ids2 = Cluster_Complete_Linkage()
counter = 1
print("\nClusters By Complete Linkage:")
for i in Ids2:
    print("Size:", len(Ids2[i]),"\tCluster",counter,": ",Ids2[i])
    counter +=1


# ------------------------PART2--------------------------
def create_Graph(file, threshold):
    data = pd.read_csv(file)
    Points = dict()
    for i in range(0,data.shape[0]):
        temp = set()
        for item in data.iloc[i,2].split('\n'):
            temp.add(item)
        Points[i]=temp
        
    # Creating a Graph    
    G = nx.Graph()
    for i in range(0,len(Points)):
        G.add_node(i)
        for j in range(i+1,len(Points)):
            x = Jaccard_Coeff(Points[i],Points[j])
            if(x >= threshold):
                G.add_edge(i, j, weight=1-x)
    # Removing edge one by one based on Betweenness Centrality
    while(nx.number_connected_components(G)<9):
        L=nx.edge_betweenness_centrality(G,normalized=True)
        temp = -math.inf
        for e in L:
            if(temp<L[e]):
                edge=e
        G.remove_edge(*edge)
    return G

def get_9_clusters(G):
    CLUSTERS = dict()
    counter = 0
    res = []
    for i in nx.connected_components(G):
        res.append((i,len(i)))
    res = sorted(res, key=lambda x:(-x[1],x[0]))
    for i in res:
        if(counter==9):
            break
        CLUSTERS[counter] = i[0]
        counter+=1
    return CLUSTERS

print("\n\nPART2:\n")
threshold = 0.279
G = create_Graph("AAAI.csv",threshold)
print("Graph when Threshold =",threshold)
# nx.draw(G, with_labels=False, node_size=20)
CLUSTERS = get_9_clusters(G)
# plt.show()
for i in CLUSTERS:
    print("Size:", len(CLUSTERS[i]),"Cluster",i+1,": ",CLUSTERS[i])

# ------------------------PART3--------------------------
def I(Clusters, Label, N):
    Total = 0.0
    for W in Clusters:
        Wk = Clusters[W]
        for C in Label:
            Cj = Label[C]
            s1 = set(Wk)
            s2 = set(Cj)
            x = len(s1.intersection(s2))
            if(x != 0):
                z = math.log((N*x)/(len(s1)*len(s2)))
                Total += (x/N)*z
    return Total
def H(W,N):
    Total = 0.0
    for w in W:
        Wk = W[w]
        s1 = set(Wk)
        x = len(s1)
        if(x != 0):
            Total += (x/N)*math.log(x/N)
    return -Total

def NMI(Clusters, Label, N):
    x = I(Clusters, Label, N)
    y = H(Clusters,N)
    z = H(Label,N)
    return (2.0*x)/(y+z)

data = pd.read_csv('AAAI.csv')
Label = dict()
for i in range(0,len(data)):
    if(data.iloc[i,3] not in Label):
        Label[data.iloc[i,3]]=[]
    Label[data.iloc[i,3]].append(i)

print("\n\nPART3:\n")
print("NMI for Hierarchical Clustering method with Single Linkage")
print(NMI(Ids1, Label, len(data)),"\n\n")

print("NMI for Hierarchical Clustering method with Complete Linkage")
print(NMI(Ids2, Label, len(data)),"\n\n")

print("NMI for Girvan-Newman clustering algorithm")
print(NMI(CLUSTERS, Label, len(data)),"\n\n")


print("NMI for Girvan-Newman clustering algorithm")
ans = -math.inf
best_thres = 0.0
L = [0.01]
for threshold in np.arange(0.999,0.001,-0.03): 
    G=create_Graph("AAAI.csv",threshold)
    CLUSTERS = get_9_clusters(G)
    x = NMI(CLUSTERS, Label, len(data))
    if(x>ans):
        best_thres = threshold
        ans = x
    print("Threshold:",round(threshold,5),"  \tBest_Threshold:", round(best_thres,5),"\tMax_NMI:",ans)

