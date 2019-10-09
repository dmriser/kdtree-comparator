''' Custom imp of kd-tree for use in binning mutlivariate samples '''

import matplotlib.pyplot as plt 
import numpy as np 

class Node:

    def __init__(self, data=None, bounds=None, depth=None, aux_data=None):
        self.data = data 
        self.aux_data = aux_data
        self.depth = depth or 0

        if data is not None and aux_data is not None:
            assert(data.shape[0] == aux_data.shape[0])

        if bounds is None:
            self.bounds = self._determine_bounds()
        else:
            self.bounds = bounds 

    def _determine_bounds(self):
        bounds = np.zeros((2, self.data.shape[1]))
        bounds[0,:] = self.data.min(axis=0)
        bounds[1,:] = self.data.max(axis=0)
        return bounds 

    def split(self):
        axis = self.depth % self.data.shape[1]
        idx = np.argsort(self.data[:,axis])
        split_idx = self.data.shape[0] // 2
        self.data = self.data[idx]

        left_bounds, right_bounds = self.bounds.copy(), self.bounds.copy()
        left_bounds[1,axis] = self.data[self.data.shape[0]//2,axis]
        right_bounds[0,axis] = self.data[self.data.shape[0]//2,axis]

        if self.aux_data is not None:
            self.aux_data = self.aux_data[idx]

            return (Node(data=self.data[:split_idx,:], bounds=left_bounds, 
                         depth=self.depth+1, aux_data=self.aux_data[:split_idx]), 
                    Node(data=self.data[split_idx:,:], bounds=right_bounds, 
                         depth=self.depth+1, aux_data=self.aux_data[split_idx:]))

        else:
            return (Node(data=self.data[:split_idx,:], bounds=left_bounds, depth=self.depth+1), 
                    Node(data=self.data[split_idx:,:], bounds=right_bounds, depth=self.depth+1))

def build(root, leafsize=100, nodes=None):
    nodes = nodes or [] 

    if root.data.shape[0] <= leafsize:
        nodes.append(root)
        return nodes

    else:
        left, right = root.split() 
        left_tree = build(left, leafsize, nodes=nodes)
        right_tree = build(right, leafsize, nodes=nodes)
        nodes.extend(left_tree)
        nodes.extend(right_tree)

    return nodes 

def get_k_worst_nodes(nodes, scores, k=3):
    
    if len(nodes) < k:
        raise ValueError('Nodes must be longer then the k-value')

    order = np.argsort(scores)[:k]
    return [nodes[i] for i in order]

def get_k_best_nodes(nodes, scores, k=3):
    
    if len(nodes) < k:
        raise ValueError('Nodes must be longer then the k-value')

    order = np.argsort(scores)[::-1][:k]
    return [nodes[i] for i in order]

def plot_k_best_nodes(nodes, scores, k=1):

    best_nodes = get_k_best_nodes(nodes, scores, k)
    dim = nodes[0].data.shape[1]

    fig, axes = plt.subplots(nrows=k, ncols=dim, figsize=(dim*4, k*3))

    for i in range(k):
        node = best_nodes[i]
        for j in range(dim):
            idx1 = np.where(node.aux_data == 0)[0]
            idx2 = np.where(node.aux_data == 1)[0]
            axes[i,j].hist(node.data[idx1,j], bins=np.linspace(node.bounds[0,j], node.bounds[1,j], 20), alpha=0.6, edgecolor='k')
            axes[i,j].hist(node.data[idx2,j], bins=np.linspace(node.bounds[0,j], node.bounds[1,j], 20), alpha=0.6, edgecolor='k')
            axes[i,j].grid(alpha=0.2)

    fig.tight_layout() 
    return fig, axes 

def plot_k_worst_nodes(nodes, scores, k=1):

    worst_nodes = get_k_worst_nodes(nodes, scores, k)
    dim = nodes[0].data.shape[1]

    fig, axes = plt.subplots(nrows=k, ncols=dim, figsize=(dim*4, k*3))

    for i in range(k):
        node = worst_nodes[i]
        for j in range(dim):
            idx1 = np.where(node.aux_data == 0)[0]
            idx2 = np.where(node.aux_data == 1)[0]
            axes[i,j].hist(node.data[idx1,j], bins=np.linspace(node.bounds[0,j], node.bounds[1,j], 20), alpha=0.6, edgecolor='k')
            axes[i,j].hist(node.data[idx2,j], bins=np.linspace(node.bounds[0,j], node.bounds[1,j], 20), alpha=0.6, edgecolor='k')
            axes[i,j].grid(alpha=0.2)

    fig.tight_layout() 
    return fig, axes 

if __name__ == '__main__':

    nsamples = 100000
    dim = 5
    #x = np.random.uniform(0, 1, (nsamples,dim))
    #y = np.random.binomial(n=1, p=0.5, size=nsamples)
    #print(x.shape)

    x1 = np.random.normal(0, 1, (nsamples, dim))
    x2 = np.random.normal(0.4, 1.2, (nsamples, dim))
    x = np.vstack([x1, x2])
    y = np.hstack([np.repeat(0, nsamples), np.repeat(1, nsamples)])

    print(x.shape, y.shape)

    root = Node(data=x, depth=0, aux_data=y)
    nodes = build(root, leafsize=200)

    for i, node in enumerate(nodes):
        print(i, node.data.shape, node.bounds, node.depth, node.aux_data.mean())


    from scipy.stats import binom_test

    scores = np.zeros(len(nodes))
    for i,node in enumerate(nodes):
        npoints = node.data.shape[0]
        nheads = len(node.aux_data[node.aux_data == 1])
        ntails = len(node.aux_data[node.aux_data == 0])
        assert(npoints == (nheads + ntails))

        # Assuming that the coin is fair, what is the probability 
        # of getting this result? 
        prob = binom_test(x=nheads, n=npoints, p=0.5)
        asym = np.abs(nheads - ntails) / (nheads + ntails)
        print(f'Points {npoints}, Heads {nheads}, Tails {ntails}, Prob {prob}, Asym {asym}')
        
        scores[i] = prob

    

    order = np.argsort(scores)[::-1]
    for index in order:
        print(f'Node {index} has proba {scores[index]}')
        

    fig, axes = plot_k_best_nodes(nodes,scores,3)
    fig.savefig('topk.pdf')

    fig, axes = plot_k_worst_nodes(nodes,scores,3)
    fig.savefig('lastk.pdf')
