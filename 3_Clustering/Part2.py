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

# ------------------------PART2--------------------------
threshold = 0.279
G = create_Graph("AAAI.csv",threshold)
print("Graph when Threshold =",threshold)
nx.draw(G, with_labels=False, node_size=20)
CLUSTERS = get_9_clusters(G)
plt.show()
for i in CLUSTERS:
    print("Size:", len(CLUSTERS[i]),"\tCluster",i+1,": ",CLUSTERS[i])