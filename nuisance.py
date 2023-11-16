"""
    Implements a simple framework for nuisance parameters and sets of them with correlated priors
"""

import numpy as np
from copy import copy

class Nuisance:
    def __init__(self, mean, std, grad, name):
        self._mean = mean
        self._std = std
        self._gradient = grad
        self._name = name

    @property
    def mean(self):
        return copy(self._mean)
    @property
    def std(self):
        return copy(self._std)
    @property
    def gradient(self):
        return self._gradient
    @property
    def name(self):
        return self._name

class NuisanceSet:
    def __init__(self) -> None:
        self._correlation = np.array([])
        self._inv = None
        self._means = np.array([])
        self._stds = np.array([])
        self._names = []

    @property
    def means(self):
        return self._means
    @property
    def stds(self):
        return self._stds
    @property
    def names(self):
        return self._names

    def add_params(self, adding_cor:np.ndarray, *params:Nuisance):
        assert len(np.shape(adding_cor))==2, "{}".format(np.shape(adding_cor))

        new_dim = len(self._correlation) + len(adding_cor)

        new_cor = np.zeros((new_dim, new_dim))
        for i in range(len(self._correlation)):
            for j in range(len(self._correlation)):
                new_cor[i][j] = self._correlation[i][j]

        for i in range(len(adding_cor)):
            for j in range(len(adding_cor)):
                new_cor[i+len(self._correlation)][j+len(self._correlation)] = adding_cor[i][j]

        self._correlation = new_cor

        self._means = np.concatenate((self._means, [entry.mean for entry in params]))
        self._stds = np.concatenate((self._stds, [entry.std for entry in params]))
        self._names += [par.name for par in params]

    def prior_penalty(self, params:np.ndarray):
        if self._inv is None:
            self._inv = np.linalg.inv(self._correlation)
        
        diffy = (params - self._means)/self._stds
        
        return 0.5*np.matmul( diffy, np.matmul(self._inv, diffy))
    
## applications follow 
