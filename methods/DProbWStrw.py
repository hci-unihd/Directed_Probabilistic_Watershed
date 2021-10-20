

from scipy.sparse import csc_matrix,eye
import numpy as np
from solvers import solvers
from graphtools.graphtools import adjacency2transition
from networkx import adjacency_matrix

import time

    
def P2PU_b(P, labeled_nodes,remapping):
    """Returns the unseeded x unseeded block and the unseeded x seeded block of the Transition matrix"""
    n=P.shape[0]
    num_labeled_nodes=len(labeled_nodes)


    # labels=sorted(np.unique(list(labeled_nodes.values())))
    
    # bT=np.zeros((n-num_labeled_nodes,len(labels)))
    # for labeled_node in labeled_nodes:
    #     label=labeled_nodes[labeled_node]
    #     reindex_labeled_node=remapping[labeled_node]
    #     bT[:,label]+=- P[reindex_labeled_node][:, num_labeled_nodes:].toarray().ravel()
    
    
    label2seed={}
    
    for labeled_node in labeled_nodes:
        label=labeled_nodes[labeled_node]
        if label in label2seed.keys():
            label2seed[label].append(remapping[labeled_node])
        else:
            label2seed[label]=[remapping[labeled_node]]
        
    b=np.zeros((n-num_labeled_nodes,len(label2seed)))
    
    for label in label2seed:
        seeds_label=label2seed[label]
        b[:,label]=(P[num_labeled_nodes:,: ][:,seeds_label].sum(1)).ravel()
        
    return csc_matrix(P[num_labeled_nodes:][:, num_labeled_nodes:]),b

def e_(i,n):
    e=np.zeros((1,n))
    e[0,i]=1
    return e
    
def pu2p(pu,labeled_nodes,no_inf_nodes=set()):
    '''
    Adds the seed nodes and the zero knowledge nodes to the probability array.
    Zero knowledge nodes form a new label indexed in the last column,
    to indicate that they are zero knowledge nodes

    Parameters
    ----------
    pu : np.array #nodesx#labels
        Assignment probability of the unseeded nodes.
    labeled_nodes : np.array
        column0=nodes, column1=labels
    no_inf_nodes : set
        set of zero knowledge nodes.

    Returns
    -------
    pu : np.array #nodesx#labels+1
        Assignment probability array of the all the nodes. 

    '''
    num_labels=pu.shape[1]+1
    pu=np.hstack((pu,np.zeros((pu.shape[0],1))))
    nodes_to_add=sorted(set(labeled_nodes.keys()).union(no_inf_nodes))
    for node in nodes_to_add:
        if node in no_inf_nodes:
            label=-1
        else:
            label=labeled_nodes[node]
        pu=np.vstack((pu[:node,:],e_(label,num_labels),pu[node:,:]))
    return pu

def obtain_approx_probabilities_trw(Pu,b,num_nodes,eta=0.01,solving_mode='direct'):
    '''
    This function solves Phat_U.T*x=bhat.T where Phat is the probability distribution
    of the TRW (teleported random walker). Since, Phat=(1-eta)P+eta*np.ones((n,1))@np.ones((n,1)).T
    where P is the original transition probability matrix of the graph.
    we use the Sherman–Morrison formula to solve the linear system without renouncing
    to the sparsity of P.

    Parameters
    ----------
    Pu : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    num_nodes : TYPE
        DESCRIPTION.
    eta : TYPE, optional
        DESCRIPTION. The default is 0.01.
    solving_mode : TYPE, optional
        DESCRIPTION. The default is 'direct'.

    Returns
    -------
    pu : TYPE
        DESCRIPTION.

    '''
    u=np.ones((Pu.shape[0],1))
    b_=np.hstack([(1-eta)*b+eta/num_nodes*u,u])
    eye_=eye(Pu.shape[0]).tolil()
    for i in np.where((Pu.toarray().sum(1)+b.sum(1))==0)[0]:
        eye_[i,i]=1-eta
    pu_=solvers[solving_mode]((eye_-(1-eta)*Pu).tocsc(), b_)
    Qu=pu_[:,-1]
    pu=(eta/num_nodes)*(np.ones((b.shape[1], 1))*Qu).T*(u.T@pu_[:,:-1]).ravel()
    pu=pu_[:,:-1]+pu/(1-Qu.sum()*(eta/num_nodes))
    return pu
    
def DProbWStrw_approx(G,labeled_nodes,eta=0.01,solving_mode='direct'):
    '''
    Implementation of the Directed Probabilistic Watershed algorithm
    when a teleported random walker (TRW) is assumed.
    
    Instead of solving L_U.T*x=b.T (with the weights that would generate the TRW),
    here it is solved Phat_U.T*x=bhat.T.T where Phat is the probability distribution
    of the TRW. The second linear system is obtained from the first just by 
    left-multiplying both sides by the D.power(-1), where D is the diagonal matrix
    formed by the diagonal of L_U.T
    
    
    Parameters
    ----------
    G : networkx graph
        
    labeled_nodes : dictionary
        key=nodes, value=labels
    eta : float, optional
        Probability teleportation. The default is 0.01.
    solving_mode : string, optional
        Different solvers of the scipy library. The default is 'direct'.
        "direct",  
        "bicgstab",
        "bicg",
        "cgs,
        "gmres,
        "lgmres",
        "qmr"

    Returns
    -------
    p : Probability assignment array
    '''
    mask_l=list(labeled_nodes.keys())
    mask_u=sorted((set(range(G.number_of_nodes())).difference((set(mask_l)))))
    A=adjacency_matrix(G,nodelist=mask_l+mask_u)
    remapping={node:i for i,node in enumerate(mask_l)}
    P=adjacency2transition(A)
    Pu,b=P2PU_b(P,labeled_nodes,remapping)
    start=time.time()
    assert (solving_mode=='direct')
    pu=obtain_approx_probabilities_trw(Pu,b,P.shape[0],eta,solving_mode)

    print('solve time=',time.time()-start)
    p = pu2p(pu, labeled_nodes)
    return p

#%%%%%%

def DProbWStrw(G,labeled_nodes,eta=0.01,solving_mode='direct'):
    '''
    Implementation of the Directed Probabilistic Watershed algorithm
    when a teleported random walker (TRW) is assumed.
    
    Instead of solving L_U.T*x=b.T (with the weights that would generate the TRW),
    here it is solved Phat_U.T*x=bhat.T.T where Phat is the probability distribution
    of the TRW. The second linear system is obtained from the first just by 
    left-multiplying both sides by the D.power(-1), where D is the diagonal matrix
    formed by the diagonal of L_U.T
    
    
    Parameters
    ----------
    G : networkx graph
        
    labeled_nodes : dictionary
        key=nodes, value=labels
    eta : float, optional
        Probability teleportation. The default is 0.01.
    solving_mode : string, optional
        Different solvers of the scipy library. The default is 'direct'.
        "direct",  
        "bicgstab",
        "bicg",
        "cgs,
        "gmres,
        "lgmres",
        "qmr"

    Returns
    -------
    p : Probability assignment array

    '''
    n=G.number_of_nodes()
    mask_l=list(labeled_nodes.keys())
    mask_u=sorted((set(range(G.number_of_nodes())).difference((set(mask_l)))))
    remapping={node:i for i,node in enumerate(mask_l)}

    A=adjacency_matrix(G,nodelist=mask_l+mask_u)
    # P=((1-eta)*adjacency2transition(A).toarray()+eta*np.ones((n,n))/n)#+eta/n
    P=((1-eta)*adjacency2transition(A).toarray()+eta*np.ones((n,n))/(n-1))-np.eye(n)*eta/(n-1)
    for i in np.where(adjacency2transition(A).toarray().sum(1)==0)[0]:
        P[i,:]=np.ones(P[i,:].shape)/n
    Pu,b=P2PU_b(P,labeled_nodes,remapping)
    start=time.time()

    pu=np.linalg.solve(np.eye(n-len(labeled_nodes))-Pu,b)
    print('solve time=',time.time()-start)
    p = pu2p(pu, labeled_nodes)
    return p

