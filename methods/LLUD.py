from graphtools.graphtools import adjacency2transition
from scipy.sparse import lil_matrix,diags
from numpy import unique
import numpy as np

def generateY(n,labeled_nodes):
    """Generates the independent term of the linear system to solve"""
    labels=unique(list(labeled_nodes.values()))
    Y=lil_matrix((n,len(labels)))
    
    for labeled_node in labeled_nodes:
        label=labeled_nodes[labeled_node]
        Y[labeled_node,label]=1
        
    return Y

def stationary_dist(A,eta):
    '''
    To trasnform Random Walker to random walker teleport with probability eta
    the adjacency matrix transforms in the following way:
        if i!=i
            aij=(1-eta)*aij+eta*d_i/(n-1)
        else:
            aii=0
    where d_i is the outdegree of node i and n is the number of nodes.
    
    The stationary is equal to  PI(v)=D(v)/sum(D) where D is the vector of 
    in-degrees in the new adjacency matrix.
    
    
    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    eta : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    n=A.shape[0]
    d=np.array(A.sum(1))
    A_=np.array(((1-eta)*A+eta*d/(n-1)))
    zero_out_deg_nodes=np.where(A_.sum(1).ravel()==0)
    A_[zero_out_deg_nodes]=np.ones((zero_out_deg_nodes[0].size,n))
    A_=A_-np.diag(np.diagonal(A_))
    PI=A_.sum(0)/A_.sum()
    return diags(PI)
    
def LLUD(A,labeled_nodes,alpha=0.1,eta=0.01):
    '''
    implementation of the algorithm in 
    "Learning from labeled and unlabeled data on a directed" by Zhou et al., ICML 2005

    Parameters
    ----------
    A : scipy sparse matrix
        adjacency matrix of the graph
    labeled_nodes : np.array
        column0=nodes, column1=labels
    alpha : float, optional
        regularization parameter. The default is 0.1.
    eta : float, optional
        Probability teleportation. The default is 0.01.


    Returns
    -------
    Aff : np.array
        #nodes x #labels array where entry (i,j) indicates the affinity of 
        the node i to the label j.

    '''
    n=A.shape[0]
    P=((1-eta)*adjacency2transition(A).toarray()+eta*np.ones((n,n))/n)+eta/(n-1)
    for i in np.where(adjacency2transition(A).toarray().sum(1)==0)[0]:
        P[i,:]=np.ones(P[i,:].shape)/n
    Y=generateY(n,labeled_nodes).toarray()
    PI=stationary_dist(A,eta)
    PHI=(PI.power(1/2)@P@PI.power(-1/2)+PI.power(-1/2)@P@PI.power(1/2))/2
    Aff=np.linalg.solve(np.eye(n)-alpha*PHI,Y)

    return Aff