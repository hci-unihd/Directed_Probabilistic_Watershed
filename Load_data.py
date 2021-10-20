#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:11:50 2021

@author: enfita
"""

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups,fetch_olivetti_faces
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn import datasets

folder='data/'
def Load_data( data='email_eu',Graphtype=nx.DiGraph(),n_neighbors=5):
    if data=='email_eu':
        filename_edges= folder+'email_EU/email-Eu-core.txt'
        filename_labels=folder+'email_EU/email-Eu-core-department-labels.txt'
        G=nx.read_edgelist(filename_edges,create_using=Graphtype,nodetype=int)
        G.remove_edges_from(nx.selfloop_edges(G))
        labels={}
        with open(filename_labels) as f:
            for row in f:
                node,label = row.split(' ')
                label=int(label.split('\n')[0])
                labels[int(node)]=label
    elif data=='20news':
        

        newsgroups_train = fetch_20newsgroups(subset='train')
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(newsgroups_train.data)
        labels={i:label for i,label in enumerate(newsgroups_train.target)}
        A = kneighbors_graph(vectors, n_neighbors,  mode='distance')
        A.data=np.exp( -A.data)
        G=nx.from_scipy_sparse_matrix(A, parallel_edges=False, create_using=Graphtype, edge_attribute='weight')
        
    elif data=='digits':
        digits = datasets.load_digits()
        labels={i:label for i,label in enumerate(digits.target)}
        A = kneighbors_graph(digits.data, n_neighbors,  mode='distance')
        A.data=np.exp( -A.data)
        G=nx.from_scipy_sparse_matrix(A, parallel_edges=False, create_using=Graphtype, edge_attribute='weight')
    elif data=='cora':
        filename_edges= folder+'cora/cora.cites'
        filename_labels=folder+'cora/cora.content'
        G=nx.read_edgelist(filename_edges,create_using=Graphtype,nodetype=int)
        mapping={node:i for i,node in enumerate(list(G.nodes))}
        labels={}
        nx.relabel_nodes(G, mapping, copy=False)
        label2int={'Case_Based':0,
                   'Genetic_Algorithms':1,
                   'Neural_Networks':2,
                   'Probabilistic_Methods':3,'Reinforcement_Learning':4,
                   'Rule_Learning':5,'Theory':6}
        with open(filename_labels) as f:
            for row in f:
                splited_row= row.split('\t')
                node=int(splited_row[0])
                label=label2int[splited_row[-1].split('\n')[0]]
                labels[mapping[int(node)]]=label
    elif data=='citeseer':
        filename_edges= folder+'citeseer/citeseer.edges'
        filename_labels=folder+'citeseer/citeseer.node_labels'
        G=nx.read_weighted_edgelist(filename_edges,create_using=Graphtype,nodetype=int,delimiter=',')
        mapping={i:i-1 for i in G.nodes()}
        nx.relabel_nodes(G, mapping, copy=False)
        labels={}
        with open(filename_labels) as f:
            for row in f:
                node,label = row.split(',')
                label=int(label.split('\n')[0])
                labels[int(node)-1]=label-1
    elif data=='olivetti':
        olivetti = fetch_olivetti_faces()
        labels={i:label for i,label in enumerate(olivetti.target)}
        A = kneighbors_graph(olivetti.data, n_neighbors,  mode='distance')
        A.data=np.exp( -A.data)
        G=nx.from_scipy_sparse_matrix(A, parallel_edges=False, create_using=Graphtype, edge_attribute='weight')
        
    return G,labels
#     if filename[-3:]=='csv':
# #        Data  = open(filename, "r", encoding='utf8')
# ##        read = csv.reader(Data)
# #        G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
# #                          nodetype=int, data=(('weight', float),))
#         G=Graphtype
#         with open(filename_edges) as f:
#             for row in f:
#                 s = row.split(' ')
#                 G.add_edge(s[0], s[1], weight=int(s[2]))
#     else:
#         if data=='epinions':
#             G=nx.read_edgelist(filename,create_using=Graphtype,nodetype=int,data=(('weight',float),('date',str)))
#         elif data=='wiki':
#             G_pos=nx.read_edgelist(filename+'_Positive.mtx',create_using=nx.MultiGraph(),nodetype=int,data=(('weight',float),),comments='%')
#             G_neg=nx.read_edgelist(filename+'_Negative.mtx',create_using=nx.MultiGraph(),nodetype=int,data=(('weight',float),),comments='%')
#             for edge in G_neg.edges():
#                 for i in range(len(G_neg.get_edge_data(edge[0],edge[1]))):
#                     G_neg.get_edge_data(edge[0],edge[1])[i]['weight']=-1
# #                G_neg.get_edge_data(edge[0],edge[1])['weight']=-1
#             G=nx.compose(G_pos,G_neg)
#             print(G.number_of_nodes())
#             print(G.number_of_edges())
#         else:
#             G=nx.read_edgelist(filename,create_using=Graphtype,nodetype=int,data=(('weight',float),))
#     if 0 not in G.nodes():
#         mapping={}
#         for i in range(G.number_of_nodes()):
#             mapping[i+1]=i
#         G=nx.relabel_nodes(G, mapping)
