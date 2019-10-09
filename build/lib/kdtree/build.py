''' Custom imp of kd-tree for use in binning mutlivariate samples '''

import numpy as np 

from . import node

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

