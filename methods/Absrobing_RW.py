from graphtools.graphtools import adjacency2transition
from scipy.sparse import eye,lil_matrix,csc_matrix
from networkx import adjacency_matrix
from numpy import unique
import numpy as np
from solvers import solvers

def generateY(n,labeled_nodes):
    """Generates the independent term of the linear system to solve"""
    labels=unique(list(labeled_nodes.values()))
    Y=lil_matrix((n,len(labels)))
    
    for labeled_node in labeled_nodes:
        label=labeled_nodes[labeled_node]
        Y[labeled_node,label]=1
        
    return csc_matrix(Y)

def absorbingRW(G,labeled_nodes,alpha=0.1,solving_mode='direct'):
    '''
    Implementation of the paper 
    
    "Transduction on Directed Graphs via Absorbing Random Walks"
    De, Jaydeep and Zhang, Xiaowei and Lin, Feng and Cheng, Li
    
    Parameters
    ----------
    G : networkx graph
        
    labeled_nodes : dictionary
        key=nodes, value=labels
        
    alpha : float, optional
        Global probability of staying within the transient nodes.

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
    Aff : np.array
        #nodes x #labels array where entry (i,j) indicates the affinity of 
        the node i to the label j.

    '''
    A=adjacency_matrix(G)
    n=G.number_of_nodes()
    P=csc_matrix(adjacency2transition(A.T))#This is equivalent to P.T in De et al paper.
    Y=generateY(n,labeled_nodes)
    Aff=solvers[solving_mode](eye(n)-alpha*P, Y)
    return Aff


def absorbingRWtrw(G,labeled_nodes,alpha=0.1,eta=0.01,solving_mode='direct'):
    '''
    This function solves (I-alpha*Phatx)=Y where Phat is the probability distribution
    of the TRW (teleported random walker). Since, Phat=(1-eta)P+eta*np.ones((n,1))@np.ones((n,1)).T
    where P is the original transition probability matrix of the graph.
    we use the Sherman–Morrison formula to solve the linear system without renouncing
    to the sparsity of P.

    Parameters
    ----------
    G : networkx graph
    labeled_nodes : dictionary
        key=nodes, value=labels
    eta : TYPE, optional
        DESCRIPTION. The default is 0.01.
    solving_mode : TYPE, optional
        DESCRIPTION. The default is 'direct'.

    Returns
    -------
    aff : np.array
        #nodes x #labels array where entry (i,j) indicates the affinity of 
        the node i to the label j.

    '''
    A=adjacency_matrix(G)
    n=G.number_of_nodes()
    P=((1-eta)*adjacency2transition(A.T).toarray()+eta*np.ones((n,n))/(n-1))-np.eye(n)*eta/(n-1)
    for i in np.where(adjacency2transition(A).toarray().sum(1)==0)[0]:
        P[i,:]=np.ones(P[i,:].shape)/n
    Y=generateY(n,labeled_nodes)
    Aff=np.linalg.solve(np.array(eye(n)-alpha*P),Y.toarray())
   
    return Aff