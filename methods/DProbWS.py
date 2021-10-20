

from scipy.sparse import csc_matrix
import numpy as np
from solvers import solvers
from networkx import adjacency_matrix,bfs_tree
from graphtools.graphtools import adjacency2laplacian


import time

def mark_0knowledge_nodes(G,labeled_nodes,check_no_inf=True):
    '''
    Finds the so called zero knowledge nodes, nodes that can not reach the
    the labeled/seed nodes via a directed path.

    Parameters
    ----------
    G : networkx graph
        
    labeled_nodes : dictionary
        key=nodes, value=labels
    Returns
    -------
    A_ : scipy sparse matrix
        adjacency matrix, whose first rows and columns index the labeled nodes.
        and rows and columns corresponding to the zero knowledge nodes have
        been removed.
    no_inf_nodes : set
        set of zero knowledge nodes.

    '''
    n=G.number_of_nodes()
    if check_no_inf:
        r='metanode'
        G.add_node(r)
        for node in labeled_nodes:
            G.add_edge(node,r)
        #descendants_at_distance(G, r, G.size())
        T=bfs_tree(G,r,reverse=True)        
        
        no_inf_nodes=set(list(G.nodes())).difference(set(list(T.nodes())))
        G.remove_node(r)
        # G_=G.copy()
        # for node in no_inf_nodes:
        #     G_.remove_node(node)
    else:
        no_inf_nodes=set()
    mask_l=list(labeled_nodes.keys())
    mask_u=sorted((set(range(n)).difference((set(mask_l).union(no_inf_nodes)))))
    A=adjacency_matrix(G,nodelist=mask_l+mask_u)
    return A,mask_l,no_inf_nodes
    
def L2LUT_bT(L, labeled_nodes,remapping):
    """ Returns the unseeded x unseeded block and the unseeded x seeded block of the laplacian."""
    n=L.shape[0]
    num_labeled_nodes=len(labeled_nodes)


    # labels=sorted(np.unique(list(labeled_nodes.values())))
    
    # bT=np.zeros((n-num_labeled_nodes,len(labels)))
    # for labeled_node in labeled_nodes:
    #     label=labeled_nodes[labeled_node]
    #     reindex_labeled_node=remapping[labeled_node]
    #     bT[:,label]+=- L[reindex_labeled_node][:, num_labeled_nodes:].toarray().ravel()
    
    
    label2seed={}
    
    for labeled_node in labeled_nodes:
        label=labeled_nodes[labeled_node]
        if label in label2seed.keys():
            label2seed[label].append(remapping[labeled_node])
        else:
            label2seed[label]=[remapping[labeled_node]]
        
    bT=np.zeros((n-num_labeled_nodes,len(label2seed)))
    
    for label in label2seed:
        seeds_label=label2seed[label]
        bT[:,label]=- (L[seeds_label][:, num_labeled_nodes:].sum(0)).ravel()
        
    return csc_matrix(L[num_labeled_nodes:][:, num_labeled_nodes:].T),bT

def e_(i,n):
    e=np.zeros((1,n))
    e[0,i]=1
    return e
    
def pu2p(pu,labeled_nodes,no_inf_nodes):
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
def DProbWS(G,labeled_nodes,solving_mode='direct',check_no_inf=True):
    '''
    Implementation of the Directed Probabilistic Watershed algorithm.

    Parameters
    ----------
    G : networkx graph
        
    labeled_nodes : dictionary
        key=nodes, value=labels

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
    A,mask_l,no_inf_nodes=mark_0knowledge_nodes(G,labeled_nodes,check_no_inf)
    remapping={node:i for i,node in enumerate(mask_l)}
    L=adjacency2laplacian(A)
    if len(no_inf_nodes)>0:
        print('no_information_labels present')
    LuT,bT=L2LUT_bT(L,labeled_nodes,remapping)
    start=time.time()
    
    pu=solvers[solving_mode](LuT, bT)

    print('solve time=',time.time()-start)
    p = pu2p(pu, labeled_nodes,no_inf_nodes)
    return p

