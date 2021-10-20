import numpy as np
from graphtools.graphtools import adjacency2degree
def e_(i,n):
    e=np.zeros((1,n))
    e[0,i]=1
    return e

def normalize_similarity_matrix(A):
    D = adjacency2degree(A)
    return D.power(-1/2)@ A * D.power(-1/2)



def add_edge_zero_outdegree_nodes(A,labeled_nodes):
    '''
    Adds outgoing edges to all the nodes for those nodes whose out-degree is 0.
    It returns the adjacency matrix after adding those edges.
    '''
    zero_outdegree_nodes=list(set(np.where(np.array(A.sum(0)).ravel()==0)[0]).difference(set(labeled_nodes[:,0])))
    A=A.tolil()
    A[:,zero_outdegree_nodes]=1/A.shape[0]
    return A.tocsr()


#%%
def fillX(X,labeled_nodes):
    '''
    Adds the seed nodes to the array.


    Parameters
    ----------
    pu : np.array #nodesx#labels
        Assignment probability of the unseeded nodes.
    labeled_nodes : array


    Returns
    -------
    pu : np.array #nodesx#labels+1
        Assignment probability array of the all the nodes. 

    '''
    num_labels=X.shape[1]
    finalX=np.hstack((X,np.zeros((X.shape[0],1))))
    for node in sorted(labeled_nodes[:,0]):
        
        l=labeled_nodes[np.where(labeled_nodes[0,:]==node),1]
        finalX=np.vstack((X[:node,:],e_(l,num_labels),X[node:,:]))
    return finalX

def GTG(A,labeled_nodes,max_it=10,tol=1e-3):
    '''
    implementation of the algorithm in 

    "Graph transduction as a non-cooperative game"  Erdem and Pelillo,
    Graph-Based Representations in Pattern Recognition 2011

    Parameters
    ----------
    A : scipy sparse matrix
        adjacency matrix of the graph
    labeled_nodes : np.array
        column0=nodes, column1=labels
    max_it : int, optional
        maximum number iterations. The default is 10.
    tol : float, optional
        convergence tolerance. If ||X-X_old||<tol the algorithm stops. The default is 1e-3.

    Returns
    -------
    X: np.array
        #nodes x #labels array where entry (i,j) indicates the affinity of 
        the node i to the label j.

    '''
    A=normalize_similarity_matrix(A)
    
    labeled_nodes=np.array(list(labeled_nodes.items()))
    num_labels=len(np.unique(labeled_nodes[:,-1]))
    num_nodes=A.shape[0]
    
    non_labeled_nodes=np.array(list(set(range(num_nodes)).difference((set(labeled_nodes[:,0])))))

    
    X=np.ones((num_nodes,num_labels))/num_labels
    
    UE_2term=np.zeros((num_nodes,num_labels))        
    for j in labeled_nodes[0,:]:
            l=labeled_nodes[np.where(labeled_nodes[0,:]==j),1]
            for i in A[:,j].nonzero()[0]:
                UE_2term[i,l]+=A[i,j]
                
                

    for i,l in labeled_nodes:
        X[i,:]=e_(l,num_labels)
        
    for _ in range(max_it):
        X_old=X.copy()
        XX=X@X.T
        UX=A.multiply(XX).tocsr()
        ue=(A@X+UE_2term)[non_labeled_nodes]
        
        # ux_2term=np.zeros((num_nodes))
        

        # for j in labeled_nodes[0,:]:
        #     l=labeled_nodes[np.where(labeled_nodes[0,:]==j),1]
        #     for i in A[:,j].nonzero()[0]:
        #         ux_2term[i]+=(A[i,j]*X[i,l]).ravel()
        
        # ux=np.array(UX[non_labeled_nodes[:,np.newaxis],non_labeled_nodes].sum(1)).ravel()+ux_2term[non_labeled_nodes]
        ux=np.array(UX[non_labeled_nodes,:].sum(1)).ravel()
        assert((ux!=0).all)
        for l in range(X.shape[1]):
            X[non_labeled_nodes[:,np.newaxis],l]=(X[non_labeled_nodes[:,np.newaxis],l].ravel()*np.divide(ue[:,l],ux)).reshape((num_nodes-labeled_nodes.shape[0],1))
        # print(X_old)
        # print(X)
        if np.linalg.norm(X-X_old)<tol:
            break
    
    return fillX(X,labeled_nodes)
