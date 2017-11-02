# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 02:16:17 2016

@author: ar988996
"""

import numpy as np
from scipy import stats, integrate
import statsmodels.sandbox.stats.multicomp as multicomp
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

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

def sm2rm(S, restart_probability=0.5):
    # input positive valued similarity matrix
    R = S.copy()
    # set diagonal values to zero
    diag_idx = np.arange(R.shape[0], dtype='int64')
    R[diag_idx,diag_idx] = 0
    # normalize
    norm_factor = np.sqrt(R.sum(0))
    R = R/norm_factor.reshape(-1,1)/norm_factor.reshape(1,-1)
    # compute relevance scores
#    R = np.eye(R.shape[0]) - restart_probability*R
    R = -restart_probability*R
    R[diag_idx,diag_idx] = R[diag_idx,diag_idx] + 1
    R = (1 - restart_probability)*np.linalg.solve(R,np.eye(R.shape[0]))
    return R
    
def sm2pm(S):
    if S.shape[0] > 5000:
        P = np.zeros(S.shape, dtype='float64')
        for j in range(P.shape[1]):
            P[:,j] = (np.argsort(np.argsort(S[:,j])) + 1)/(S.shape[0] + 1)
        return P
    else:
        return (np.argsort(np.argsort(S, 0), 0) + 1)/(S.shape[0] + 1)
    
def pm2clrm(P):
    if P.shape[0] > 5000:
        Z = np.zeros(P.shape, dtype='float64')
        for j in range(Z.shape[1]):
            Z[:,j] = stats.norm.ppf(P[:,j])
        Z = (Z + Z.T)/np.sqrt(2)
        for j in range(Z.shape[1]):
            Z[:,j] = stats.norm.cdf(Z[:,j])
        return Z
    else:
        Z = stats.norm.ppf(P)
        return stats.norm.cdf((Z + Z.T)/np.sqrt(2))

def sm2clrm(S):
    return pm2clrm(sm2pm(S))

def sm2cdfm(S):
    P = ((np.argsort(np.argsort(S.reshape(-1))) + 1)/(S.size + 1)).reshape(S.shape)
#    P = ((stats.rankdata(S.reshape(-1)))/(S.size + 1)).reshape(S.shape)
    return 0.5*(P + P.T)

def quantilenormalization(X, axis):
    Xqn = np.zeros(X.shape, dtype='float64')
    if axis == 0:
        si = np.argsort(X, 0)
        ref = np.sort(X, 0).mean(1)
        si = np.argsort(si, 0)
        for i in range(X.shape[1]):
            Xqn[:,i] = ref[si[:,i]]
    elif axis == 1:
        si = np.argsort(X, 1)
        ref = np.sort(X, 1).mean(0)
        si = np.argsort(si, 1)
        for i in range(X.shape[0]):
            Xqn[i,:] = ref[si[i,:]]
    else:
        raise ValueError('axis must be 0 or 1')
    return Xqn

def robustquantilenormalization(X, axis):
    Xqn = np.zeros(X.shape, dtype='float64')
    if axis == 0:
        si = np.argsort(X, 0)
        ref = np.median(np.sort(X, 0), 1)
        si = np.argsort(si, 0)
        for i in range(X.shape[1]):
            Xqn[:,i] = ref[si[:,i]]
    elif axis == 1:
        si = np.argsort(X, 1)
        ref = np.median(np.sort(X, 1), 0)
        si = np.argsort(si, 1)
        for i in range(X.shape[0]):
            Xqn[i,:] = ref[si[i,:]]
    else:
        raise ValueError('axis must be 0 or 1')
    return Xqn

def kernel_density_estimate(X, lowerbound=None, upperbound=None, numpoints=1000, kernel='gaussian', log10range=2, numbandwidthsperlog=100, numfolds=10):
    # this doesn't seem to be working properly    
    if X.size < 2*numfolds:
#        numfolds = np.round(X.size/2).astype('int64')
        numfolds = X.size
    log10_silverman_bandwidth = np.log10((4./3./X.size)**(1./5.)*X.std())
    bandwidths = np.logspace(log10_silverman_bandwidth - log10range/2., log10_silverman_bandwidth + log10range/2., numbandwidthsperlog*log10range)
    renorm_pdf = False
    if lowerbound == None:
        lowerbound = X.min() - 0.5*(X.max() - X.min())
        Xl = np.zeros(0, dtype='float64')
    else:
        renorm_pdf = True
        Xl = 2*lowerbound - X
#        Xl = np.zeros(0, dtype='float64')
    if upperbound == None:
        upperbound = X.max() + 0.5*(X.max() - X.min())
        Xu = np.zeros(0, dtype='float64')
    else:
        renorm_pdf = True
        Xu = 2*upperbound - X
#        Xu = np.zeros(0, dtype='float64')
    X = np.concatenate((Xl, X, Xu), 0)
    x = np.linspace(lowerbound, upperbound, numpoints, dtype='float64')
    gridsearch = GridSearchCV(KernelDensity(kernel=kernel), param_grid={'bandwidth':bandwidths}, cv=numfolds)
    gridsearch.fit(X[:,np.newaxis])
    print('bandwidth = {!s}'.format(gridsearch.best_params_['bandwidth']))
    print('numfolds = {!s}'.format(gridsearch.n_splits_))
    print('numpoints = {!s}'.format(X.size))
    if gridsearch.best_params_['bandwidth'] == bandwidths.min():
        print('cross-validation failed to find bandwidth in range. best score at minimum bandwidth')
        success = False
    elif gridsearch.best_params_['bandwidth'] == bandwidths.max():
        print('cross-validation failed to find bandwidth in range. best score at minimum bandwidth')
        success = False
    else:
        print('cross-validation succeed. best score at bandwidth in range.')
        success = True
    test_score_mean = gridsearch.cv_results_['mean_test_score']
    test_score_std = gridsearch.cv_results_['std_test_score']
    train_score_mean = gridsearch.cv_results_['mean_train_score']
    train_score_std = gridsearch.cv_results_['std_train_score']
    plt.figure()
    plt.errorbar(np.log10(bandwidths), test_score_mean, yerr=test_score_std, fmt='-k')
    plt.errorbar(np.log10(bandwidths), train_score_mean, yerr=train_score_std, fmt='-r')
    plt.show()
#    plt.close()
    kde = gridsearch.best_estimator_
    pdf = np.exp(kde.score_samples(x[:,np.newaxis]))
    if renorm_pdf:
        pdf = pdf/np.trapz(pdf, x)
    plt.figure()
    plt.plot(x, pdf, '-k')
    plt.show()
#    plt.close()
    return pdf, x, success

def pdf2pmf(pdf, x):
    cdf = integrate.cumtrapz(pdf, x, initial=0)
    pmf = np.diff(cdf)
    xc = x[:x.size-1] + np.diff(x)/2.
    return pmf, xc
    
def sample_pmf(pmf, x, numsamples, replace=True):
    return np.random.choice(x, numsamples, replace=replace, p=pmf)

#def kde_smoothed_bootstrap(X, lowerbound=None, upperbound=None, numbootstraps=10**5, statistic=np.mean, numsmoothedpoints=1000, kernel='gaussian', log10range=2, numbandwidthsperlog=100, numfolds=10):
#    # kernel_density_estimate doesn't seem to be working properly
#    pdf, x, kde_success = kernel_density_estimate(X, lowerbound=lowerbound, upperbound=upperbound, numpoints=numsmoothedpoints+1, kernel=kernel, log10range=log10range, numbandwidthsperlog=numbandwidthsperlog, numfolds=numfolds)
#    pmf, x = pdf2pmf(pdf, x)
#    bootstrap_statistic = np.zeros(numbootstraps, dtype='float64')
#    for i in range(numbootstraps):
#        bootstrap_statistic[i] = statistic(sample_pmf(pmf, x, numsamples=X.size, replace=True))
#    if lowerbound == None:
#        lowerbound = bootstrap_statistic.min()
#    if upperbound == None:
#        upperbound = bootstrap_statistic.max()
#    counts, bin_edges = np.histogram(bootstrap_statistic, max(10, np.round(numbootstraps/1000).astype('int64')), range=(lowerbound,upperbound))
#    probs = counts/counts.sum()
#    bins = bin_edges[:bin_edges.size-1] + np.diff(bin_edges)/2.
#    plt.figure()
#    plt.bar(bins, probs, bins[1]-bins[0])
#    plt.show()
#    plt.close()
#    return bootstrap_statistic, probs, bins

def smoothed_bootstrap(X, lowerbound=None, upperbound=None, numbootstraps=10**5, statistic=np.mean):
    silverman_bandwidth = (4./3./X.size)**(1./5.)*X.std() # ~ 1/np.sqrt(X.size)
    bootstrap_statistic = np.zeros(numbootstraps, dtype='float64')
    for i in range(numbootstraps):
        bootstrap_statistic[i] = statistic(np.random.choice(X, X.size, replace=True) + np.random.randn(X.size)*silverman_bandwidth)
    if lowerbound == None:
        lowerbound = bootstrap_statistic.min()
    if upperbound == None:
        upperbound = bootstrap_statistic.max()
    counts, bin_edges = np.histogram(bootstrap_statistic, max(10, np.round(numbootstraps/1000).astype('int64')), range=(lowerbound,upperbound))
    probs = counts/counts.sum()
    bins = bin_edges[:bin_edges.size-1] + np.diff(bin_edges)/2.
    plt.figure()
    plt.bar(bins, probs, bins[1]-bins[0])
    plt.show()
    plt.close()
    return bootstrap_statistic, probs, bins

def bootstrap(X, lowerbound=None, upperbound=None, numbootstraps=10**5, statistic=np.mean):
    bootstrap_statistic = np.zeros(numbootstraps, dtype='float64')
    for i in range(numbootstraps):
        bootstrap_statistic[i] = statistic(np.random.choice(X, X.size, replace=True))
    if lowerbound == None:
        lowerbound = bootstrap_statistic.min()
    if upperbound == None:
        upperbound = bootstrap_statistic.max()
    counts, bin_edges = np.histogram(bootstrap_statistic, max(10, np.round(numbootstraps/1000).astype('int64')), range=(lowerbound,upperbound))
    probs = counts/counts.sum()
    bins = bin_edges[:bin_edges.size-1] + np.diff(bin_edges)/2.
    plt.figure()
    plt.bar(bins, probs, bins[1]-bins[0])
    plt.show()
    plt.close()
    return bootstrap_statistic, probs, bins
