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

# ------------------------PART1--------------------------

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