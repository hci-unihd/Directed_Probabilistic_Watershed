import numpy as np
import matplotlib.pyplot as plt



def sampling_labeled_data(G,labels,percentage_labeled_data,mode='uniform_per_class'):
    '''
    

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    labels : array
    percentage_labeled_data : float
        percentage of nodes that will be sampled.
    mode : string, optional
        default is uniform_per_class
        If uniform_per_class: it samples uniformly from each label set a percentage of the 
        the nodes of each class according to the parameter percentage_labeled_data
        If uniform_total: it samples a percentage of the nodes according to the parameter
        percentage_labeled_data ignoring the classes
        If prop_indeg: it samples a percentage of the 
        the nodes of each class according to the parameter percentage_labeled_data
        with probability proportional to its in-degree

    Returns
    -------
    labeled_nodes_ : TYPE
        DESCRIPTION.

    '''
    
    number_of_nodes=G.number_of_nodes()
    if mode=='uniform_per_class':
        
        nodes_sorted_by_label={}
        for node,label in labels.items():
            if label in nodes_sorted_by_label.keys():
                nodes_sorted_by_label[label].append(node)
            else:
                nodes_sorted_by_label[label]=[node]
                
        labeled_nodes_={}
        for label in nodes_sorted_by_label:
            size=np.int(np.ceil(percentage_labeled_data*len(nodes_sorted_by_label[label])))
            nodes_=np.random.choice(nodes_sorted_by_label[label],size=size,replace=False)
            for node in nodes_:
                labeled_nodes_[node]=label

    elif mode=='uniform_total':
        nodes_=np.random.choice(list(labels.keys()),size=np.int(percentage_labeled_data*number_of_nodes),replace=False)
        labeled_nodes_={node:labels[node] for node in nodes_}
    elif mode=='prop_indeg':
        nodes_sorted_by_label={}
        probs_sorted_by_label={}
        for node,label in labels.items():
            if label in nodes_sorted_by_label.keys():
                nodes_sorted_by_label[label].append(node)
                probs_sorted_by_label[label].append(G.in_degree(node))
            else:
                nodes_sorted_by_label[label]=[node]
                probs_sorted_by_label[label]=[G.in_degree(node)]

                
        labeled_nodes_={}
        for label in nodes_sorted_by_label:
            size=np.int(np.ceil(percentage_labeled_data*len(nodes_sorted_by_label[label])))
            prob=np.array(probs_sorted_by_label[label])+1e-8
            nodes_=np.random.choice(nodes_sorted_by_label[label],size=size,replace=False,p=prob/sum(prob))
            for node in nodes_:
                labeled_nodes_[node]=label
    return labeled_nodes_




def symmetry(G):
    """Computes how many edges in the graph are present in both directions"""
    sym=0
    for e in G.edges():
        u,v=e
        if (v,u) in G.edges():
            sym+=1
    
    print('% of edges that are symmetrized=',sym/G.number_of_edges()*100)
    


def Accuracy(prediction,true_labels,num_classes):
    '''
    Computes the accuracy score of a classifier given its prediction
    couting the zero-knowledge nodes (nodes whose label can not be inferred): 
        acc=total_correctly_predicted/total_data
    and ignoring the zero-knowledge nodes: 
        acc_0_knowkledge_nodes=(total_correctly_predicted-total_0_knowledges_nodes)/(total_data-total_0_knowledges_nodes)
    Parameters
    ----------
    prediction : iterable
        predicted labels 
    true_labels : iterable
        true labels 
    num_classes : int, DEFAULT: None
        number of classes. The possible classes are assumed to range
        from 0 to num_classes-1
        
        Zero_knowledge nodes are assigned the class num_classes.

    Returns
    -------
    accuracy : 
        Number of  correctly labeled divided by total number of nodes.
    accuracy_without_0_knowledges_nodes : 
        Number of  correctly labeled nodes divided by total number of nodes minus the zero knowledge nodes
        
    total_0_knowledges_nodes : 
        Number of nodes that are zero knowledge nodes

    '''
    total_counted=0
    total_0_knowledges_nodes=0
    total_right=0
    for node in true_labels:
        if prediction[node]==num_classes:
            total_0_knowledges_nodes+=1
        elif true_labels[node]==prediction[node]:
            total_right+=1
        total_counted+=1
    
    accuracy_without_0_knowledges_nodes=total_right/(total_counted-total_0_knowledges_nodes)
    accuracy=total_right/(total_counted)
    print('Accuracy=%0.2f, Accuracy_0_knowkledge_nodes=%0.2f, %% Zero-knowledge nodes=%0.2f%%'%(accuracy,accuracy_without_0_knowledges_nodes,total_0_knowledges_nodes/total_counted*100))
    return accuracy,accuracy_without_0_knowledges_nodes,total_0_knowledges_nodes


def plot_evolution(dictionary,label):
        
    x = dictionary.keys()
    y = [dictionary[i][0] for i in dictionary]
    e = [dictionary[i][1] for i in dictionary]
    if label=='':
        plt.errorbar(x, y, e,marker='o')
    else:
        plt.errorbar(x, y, e, label=label,marker='o')
