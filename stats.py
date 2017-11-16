# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import numpy as np
import statsmodels.sandbox.stats.multicomp as multicomp
from scipy import stats

def mad(X, axis, keepdims=False):
    return 1.4826*np.median(np.abs(X - np.median(X, axis=axis, keepdims=True)), axis=axis, keepdims=keepdims)

def corr_pearson(X, Y=np.zeros(0), axis=0, getpvalues=False):
    if axis == 0:
        X = X.T
    elif axis == 1:
        pass
    else:
        raise ValueError('invalid axis')
    X = X - X.mean(axis=0)
    X = X/np.linalg.norm(X, ord=2, axis=0) # same as X = X/np.sqrt((X**2).sum(axis=0))
    n = X.shape[0]
    if Y.size == 0:
        r = (X.T).dot(X)
    else:
        if axis == 0:
            Y = Y.T
        Y = Y - Y.mean(axis=0)
        Y = Y/np.linalg.norm(Y, ord=2, axis=0)
        r = (X.T).dot(Y)
    if getpvalues:
        p = 2*stats.norm.cdf(-np.abs((np.log((np.float64(1) + r)/(np.float64(1) - r))/np.float64(2))*np.sqrt(np.float64(n) - np.float64(3))))
        return r, p
    else:
        return r
    
def corr_spearman(X, Y=np.zeros(0), axis=0, getpvalues=False):
    if axis == 0:
        X = X.T
    elif axis == 1:
        pass
    else:
        raise ValueError('invalid axis')
    X = np.apply_along_axis(stats.rankdata, 0, X, method='average')
    if Y.size > 0:
        if axis == 0:
            Y = Y.T
        Y = np.apply_along_axis(stats.rankdata, 0, Y, method='average')
    return corr_pearson(X, Y, axis=1, getpvalues=getpvalues)
    
def corr_cosine(X, Y=np.zeros(0), axis=0, getpvalues=False):
    if axis == 0:
        X = X.T
    elif axis == 1:
        pass
    else:
        raise ValueError('invalid axis')
    X = X/np.linalg.norm(X, ord=2, axis=0) # same as X = X/np.sqrt((X**2).sum(axis=0))
    if Y.size == 0:
        r = (X.T).dot(X)
    else:
        if axis == 0:
            Y = Y.T
        Y = Y/np.linalg.norm(Y, ord=2, axis=0)
        r = (X.T).dot(Y)
    if getpvalues:
        print('Warning: p-values are not calculated for cosine similarity. Returning p=None.')
        p = None
        return r, p
    else:
        return r

def corr(X, Y=np.zeros(0), axis=0, metric='pearson', getpvalues=False):
    metric_function = {'pearson':corr_pearson, 'spearman':corr_spearman, 'cosine':corr_cosine}
    if metric not in metric_function:
        raise ValueError('invalid correlation metric')
    else:
        return metric_function[metric](X, Y, axis, getpvalues)

def fisherexacttest(x, M, n, N):
    '''
    M = size of population (e.g. genome)
    n = number of items in population with property of interest (e.g. reference gene set size)
    N = number of items drawn without replacement from population (e.g. query gene set size)
    x = number of items drawn having property of interest (e.g. overlap of gene sets)
    returns -log10(p) where p is the probability of drawing greater than or equal to x items with property of interest
    '''
    return -stats.hypergeom.logsf(x, M, n, N, loc=1)/np.log(10) # loc=1 to get p >= x

def multiple_hypothesis_testing_correction(pvalues, alpha=0.05, method='fdr_bh'):
    is_ok = ~np.isnan(pvalues)
    is_significant = np.nan*np.zeros(pvalues.shape, dtype='float64')
    pvalues_corrected = np.nan*np.zeros(pvalues.shape, dtype='float64')
    is_significant[is_ok], pvalues_corrected[is_ok] = multicomp.multipletests(pvalues[is_ok], alpha=alpha, method=method)[:2]
    is_significant = is_significant==1
    return is_significant, pvalues_corrected
