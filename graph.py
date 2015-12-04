"""
# minimal graph class to find the shortest path between two points (dijkstra)
#
# Copyright (c) 2015 Antoine Letouzey antoine.letouzey@gmail.com
# Author: Antoine Letouzey
# LICENSE: LGPL
"""

import numpy as np

def getLowestW(nodes, weights):
    n = nodes[0]
    w = weights[n]
    for i in xrange(1,len(nodes)):
        if weights[nodes[i]] < w:
            w = weights[nodes[i]]
            n = nodes[i]
    return n

# TODO : prevent jumping twice from the same place 
# 1 -> 12 and 12 -> 30 should give 1 -> 30 and not 1 -> 12 -> 30

class Graph(object):
    def __init__(self, adjMat):
        # adjMat is a square numpy matrix object
        self.adj = adjMat
        self.N = self.adj.shape[0]

    def getChild(self, n):
        tmp = list(self.adj[n,:])
        res = [i for i in range(len(tmp)) if tmp[i] != 0]
        return res
    
    def path(self, s, e):
        # init weigths and parents
        W = [np.inf for i in xrange(self.N)]
        W[s] = 0
        P = [-1 for i in xrange(self.N)]
        notSeen = range(self.N)
        while len(notSeen) > 0:
            # get unseen node with lower weight
            n1 = getLowestW(notSeen, W)
            notSeen.remove(n1)
            for n2 in self.getChild(n1):
                if W[n2] > W[n1] + 1 :
                    W[n2] = W[n1] + 1
                    P[n2] = n1
        res = []
        n = e
        while n != s:
            res.insert(0,n)
            n = P[n]
        res.insert(0,s)
        return res
        

