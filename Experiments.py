#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:32:40 2021

@author: enfita
"""
import os
from Load_data import Load_data
from methods.DProbWS import DProbWS
import numpy as np
from tools import Accuracy,sampling_labeled_data,plot_evolution
from methods.Absrobing_RW import absorbingRW,absorbingRWtrw
import matplotlib.pyplot as plt
import networkx as nx
from methods.Game_theoretic import GTG
from methods.LLUD import LLUD
from scipy import sparse
from methods.DProbWStrw import DProbWStrw

import matplotlib


#%%

uniform_sampling='uniform_per_class'#'prop_indeg'#'uniform_per_class'

folder='outputs/comparison_methods/'


solving_mode='direct'
num_reps=20
data='20news'
# data='email_eu'
# data='digits'
# data='olivetti'
# data='cora'
# data='citeseer'

eta=1e-2

if 'digits'==data:
    eta_LLUD=1e-6
    eta=1e-6

else:
    eta_LLUD=1e-2
    eta=1e-2


G,labels=Load_data(data,n_neighbors=5)
A=nx.to_scipy_sparse_matrix(G)


n=G.number_of_nodes()
num_labels=len(np.unique(list(labels.values())))
np.random.seed(21)






accuracies_DPW={}
accuracies_DPWtrw={}


accuracies_ARW={}
accuracies_ARWtrw={}

accuracies_GTG={}



accuracies_LLUD={}


no_inf_perc={}



for percentage_labeled_data in np.linspace(0.1,0.9,9):
    
    accuracies_ARW_aux=[]
    accuracies_ARWtrw_aux=[]

    accuracies_DPW_aux=[]
    accuracies_DPWtrw_aux=[]


    accuracies_GTG_aux=[]
   
    accuracies_LLUD_aux=[]



    no_inf_aux_perc=[]

    for rep in range(num_reps):
        print('percentatge=',percentage_labeled_data)
        labeled_nodes_=sampling_labeled_data(G,labels,percentage_labeled_data,mode=uniform_sampling)

        labels_present=sorted(np.unique(list(labeled_nodes_.values())))
        
        remapping_labels={i:label for i,label in enumerate(labels_present)}
        labeled_nodes={}
        for node in labeled_nodes_:
            labeled_nodes[node]=labels_present.index(labeled_nodes_[node])
        
        no_label=max(labeled_nodes.values())+1
        

        
        P=DProbWS(G,labeled_nodes,solving_mode=solving_mode)
        assignment_DPW=np.argmax(P,1)

        
        Ptrw=DProbWStrw(G,labeled_nodes,eta,solving_mode=solving_mode)
        assignment_DPWtrw=np.argmax(Ptrw,1)

        
        X=GTG(A,labeled_nodes)
        assignment_GTG=np.argmax(X,1)
        

        F=LLUD(A,labeled_nodes,alpha=0.1,eta=eta_LLUD)
        assignment_LLUD=np.argmax(F,1)
        
        

        Aff=absorbingRW(G.reverse(),labeled_nodes,alpha=0.1,solving_mode=solving_mode)#Since ARW transposes the edges, the input is the Graph transposed such that ARW undoes this transposition
        assignment_ARW=np.argmax(Aff,1) 
        #Assigns zero-knowledge label for ARW assignment
        for i in range(n):
            if assignment_DPW[i]<no_label:
                assignment_DPW[i]=remapping_labels[assignment_DPW[i]]
                assignment_ARW[i]=remapping_labels[assignment_ARW[i]]
            else:
                assignment_DPW[i]=num_labels
                assignment_ARW[i]=num_labels
        
        
        Afftrw=absorbingRWtrw(G.reverse(),labeled_nodes,alpha=0.1,eta=eta,solving_mode=solving_mode)
        assignment_ARWtrw=np.argmax(Afftrw,1)  
       
        print('DPW')
        acc_DPW,acc_w_ign_DPW,no_inf=Accuracy(assignment_DPW,labels,num_labels)
        print('DPWtrw')
        acc_DPWtrw,acc_w_ign_DPWtrw,no_inf_=Accuracy(assignment_DPWtrw,labels,num_labels)
        
        print('ARW')
        acc_ARW,acc_w_ign_ARW,no_inf_=Accuracy(assignment_ARW,labels,num_labels)
        
        
        print('ARWtrw')
        acc_ARWtrw,acc_w_ign_ARWtrw,no_inf_=Accuracy(assignment_ARWtrw,labels,num_labels)
        
        print('GTG')
        acc_GTG,acc_w_ign_GTG,no_inf_=Accuracy(assignment_GTG,labels,num_labels)
        
        print('LLUD')
        acc_LLUD,acc_w_ign_LLUD,no_inf_=Accuracy(assignment_LLUD,labels,num_labels)
        
        
        accuracies_DPW_aux.append(acc_DPW)
        accuracies_DPWtrw_aux.append(acc_DPWtrw)





        accuracies_ARW_aux.append(acc_ARW)
        accuracies_ARWtrw_aux.append(acc_ARWtrw)



        accuracies_GTG_aux.append(acc_GTG)

        accuracies_LLUD_aux.append(acc_LLUD)
       

        no_inf_aux_perc.append(no_inf/G.number_of_nodes()*100)
        
    accuracies_DPW[percentage_labeled_data]=(np.mean(accuracies_DPW_aux),np.std(accuracies_DPW_aux))
    accuracies_DPWtrw[percentage_labeled_data]=(np.mean(accuracies_DPWtrw_aux),np.std(accuracies_DPWtrw_aux))
    

    
    accuracies_ARW[percentage_labeled_data]=(np.mean(accuracies_ARW_aux),np.std(accuracies_ARW_aux))
    accuracies_ARWtrw[percentage_labeled_data]=(np.mean(accuracies_ARWtrw_aux),np.std(accuracies_ARWtrw_aux))
   
    accuracies_GTG[percentage_labeled_data]=(np.mean(accuracies_GTG_aux),np.std(accuracies_GTG_aux))
    
    accuracies_LLUD[percentage_labeled_data]=(np.mean(accuracies_LLUD_aux),np.std(accuracies_LLUD_aux))
    


    no_inf_perc[percentage_labeled_data]=(np.mean(no_inf_aux_perc),np.std(no_inf_aux_perc))
    
#%%



w,h=matplotlib.figure.figaspect(0.62)
fig, ax = plt.subplots(figsize=(2*w,2*h))


plot_evolution(accuracies_ARW,'ARW')
plot_evolution(accuracies_ARWtrw,'ARWtrw')

plot_evolution(accuracies_GTG,'GTG')


plot_evolution(accuracies_LLUD,'LLUD')

plot_evolution(accuracies_DPW,'DProbWS')
plot_evolution(accuracies_DPWtrw,'DProbWStrw')


plt.legend(loc='lower right')

# plt.yscale('log')
plt.ylabel("Accuracy")
plt.xlabel("Fraction labeled data")
plt.rcParams.update({'font.size': 32})
plt.savefig(folder+'plot_comparison_methods_%s_%ireps'%(data,num_reps))
plt.title('dataset: %s'%data)


