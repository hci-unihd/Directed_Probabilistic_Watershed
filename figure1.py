#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 22:59:44 2021

@author: enric
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from methods.DProbWS import DProbWS
from graphtools.graphtools import adjacency2laplacian
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable


def check_incoming_tree(T,seed):
    if T.number_of_nodes()!=(T.number_of_edges()+1):
        return False
    for node in T.nodes():
        if node==seed:
            if T.out_degree(node)!=0:
                # print('the seed')
                return False
        elif T.out_degree(node)!=1:
            # print('the node',node)
            return False
    T_undir=T.to_undirected()
    if not nx.is_tree(T_undir):
        return False
    
    return True

#%%
pos={0:(0,0),1:(1,0),2:(2,0),3:(0,1),4:(1,1),5:(2,1),6:(0,2),7:(1,2),8:(2,2),}

A=np.zeros((9,9))
A[0,1]=1
A[0,3]=0.5

A[1,0]=0.9
A[1,2]=0.3
A[1,4]=0.5

A[2,1]=0
A[2,5]=0.5

A[3,0]=0.2
A[3,4]=1
A[3,6]=0.

A[4,1]=0.7
A[4,3]=0.8
A[4,5]=0.5
A[4,7]=0.2

A[5,2]=0.1
A[5,4]=1
A[5,8]=0.5#Special

A[6,3]=0.5
A[6,7]=0.4

A[7,6]=0
A[7,8]=0.8

A[8,5]=0.3
#A=np.exp(np.round(np.log(A),2))
################################
w,h=matplotlib.figure.figaspect(1)
fig, ax = plt.subplots(figsize=(4*w,4*h))
plt.imshow(A)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.2)
# plt.colorbar(im,cax=cax)
plt.colorbar(fraction=0.046, pad=0.04)
plt.gca().invert_yaxis()
plt.title('Adjacency matrix')
plt.axis('off')
# plt.rcParams.update({'font.size': 80})
plt.savefig('Adjacency_matrix.png', dpi=400)



G=nx.from_numpy_matrix(A,create_using=nx.DiGraph())
G.edges(data=True)
P=DProbWS(G,{0:0,8:1})
seeds=[0,8]

for i in range(len(seeds)):
	print('Probability being connected to seed ', i+1)
	plt.figure()
	plt.title('Probability being connected to seed '+str(i+1))
	plt.imshow(np.dot(P[:,i].reshape(3,3).T,np.array([[0,0,1],[0,1,0],[1,0,0]])).T,cmap='bwr')
	plt.colorbar()
	plt.show()
plt.figure()
plt.title('Segmentation')
assignment=np.argmax(P,1).reshape((3,3))
plt.imshow(np.dot(assignment,np.array([[0,0,1],[0,1,0],[1,0,0]])).T,cmap='jet')
#plt.imshow(np.argmax(final_P,axis=2),cmap='jet')


print(assignment)

def prob2color(p):
    if p>0.5:
        return 2*((p-0.5)*np.array([255,255,255])+(1-p)*np.array([255,127.5,127.5]))
    else:
        return 2*((p)*np.array([255,255,255])+(0.5-p)*np.array([127.5,214.5,247]))
    
G_aux=G.copy()
G_aux.add_node('meta')
G_aux.add_weighted_edges_from([(8,'meta',1000),(0,'meta',1000)])

T=nx.maximum_spanning_arborescence(G_aux.reverse())

print(T.reverse().edges())

#%%
#COMPUTE FORESTS

A_=A!=0

A_bar=A_.copy()
A_bar[8,0]=1
L=adjacency2laplacian(A_)
L_bar=adjacency2laplacian(A_bar)

num_forests=np.linalg.det(L_bar[1:,1:])-np.linalg.det(L[1:,1:])
np.linalg.det(L_bar[:-1,:-1])-np.linalg.det(L[:-1,:-1])


edges={0:(0,1),1:(0,3),2:(1,0),3:(1,2),4:(1,4),5:(3,0),6:(3,4)
       ,7:(4,1),8:(4,3),9:(4,5),10:(4,7),11:(5,2),12:(5,4),13:(5,8),
       14:(6,3),15:(6,7),16:(8,5)}

edges_extra={k:v for k,v in edges.items()}
edges_extra[17]=(2,5)
edges_extra[18]=(7,8)

G_base=nx.DiGraph()
G_base.add_edges_from([(8,'meta'),(0,'meta')])
valid_forests=[]
weight_forests=[]
for combin in combinations(edges_extra.keys(),9-G_base.number_of_edges()):
    T=G_base.copy()
    T.add_edges_from([edges_extra[k] for k in combin])
    if check_incoming_tree(T,seed='meta'):
        T.remove_node('meta')
        valid_forests.append(T)
        weights=[A[e[0]][e[1]] for e in T.edges() if 'meta' not in e]
        weight_forests.append(np.prod(weights))
print(len(valid_forests))  
weight_forests_=weight_forests.copy()
# plt.figure()
# nx.draw(T,pos=pos)
# nx.draw_networkx_labels(T,pos=pos)
#%%forests=np.load('Forest_paralel_3_grid_(0,2)_(2,0).npy')
mu=1
cost_forests=-np.log(weight_forests_)
weight_forests=np.exp(-mu*cost_forests)
prob_forest=weight_forests/np.sum(weight_forests)
order=np.argsort(cost_forests)

skip_ord=[]
i=0

plt.figure()
idx_for=35
nx.draw(valid_forests[idx_for],pos=pos)
nx.draw_networkx_labels(valid_forests[idx_for],pos=pos)



w,h=matplotlib.figure.figaspect(0.4)
fig, ax = plt.subplots(figsize=(w,h))


ax.plot(cost_forests[order], prob_forest[order],color='red')
ax.bar(cost_forests[order], prob_forest[order],color='lime',width=0.01)


plt.rcParams.update({'font.size': 15})
plt.ylabel("Probability")
plt.xlabel("Cost")

# plt.tight_layout()
# ax.set_xticks(list(ax.get_xticks())[1:] + [1.5])
# plt.savefig('Probability_cost_mu=%i.png'%mu, dpi=400)
