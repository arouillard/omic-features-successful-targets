# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import numpy as np
import sklearn.feature_selection as feature_selection
import statsmodels.sandbox.stats.multicomp as multicomp
import scipy.stats as stats

def phi_coef(x, M, n, N):
#    M = size of population (e.g. genome)
#    n = number of items in population with property of interest (e.g. reference gene set size)
#    N = number of items drawn without replacement from population (e.g. query gene set size)
#    x = number of items drawn having property of interest (e.g. overlap of gene sets)
#    returns correlation coefficient for a 2x2 contingency table, i.e. for two binary arrays
#    equivalent to matthews correlation coefficient
    return (M*x - n*N)/np.sqrt(n*N*(M-n)*(M-N))
    
def fisher_exact_test(x, M, n, N):
#    M = size of population (e.g. genome)
#    n = number of items in population with property of interest (e.g. reference gene set size)
#    N = number of items drawn without replacement from population (e.g. query gene set size)
#    x = number of items drawn having property of interest (e.g. overlap of gene sets)
#    returns the probability of drawing greater than or equal to x items with property of interest
    return stats.hypergeom.sf(x, M, n, N, loc=1) # loc=1 to get p >= x

def univariate_vectorized_fisherexacttest(X, Y):
    # returns (phicoef, pvalues)
    if Y.dtype != 'float64':
        Y = Y.astype('float64')
    if X.dtype != 'float64':
        X = X.astype('float64')
    M = np.float64(Y.shape[0])
    n = X.sum(0).reshape(-1,1)
    N = Y.sum(0).reshape(1,-1)
    x = (X.T).dot(Y)
    return phi_coef(x, M, n, N), fisher_exact_test(x, M, n, N)
    
def univariate_permtest(X, Y, numperm=10**5, statistic=np.mean, tail='both'):
    # returns (difference in statistic between pos and neg classes, pvalues)
    if Y.dtype != 'bool':
        Y = Y==1
    if tail == 'left': # how likely is it to see differences as small as observed
        count_function = lambda d_shuffle, d_obs: d_shuffle <= d_obs
    elif tail == 'right': # how likely is it to see differences as large as observed
        count_function = lambda d_shuffle, d_obs: d_shuffle >= d_obs
    elif tail == 'both': # how likely is it to see differences as extreme as observed
        count_function = lambda d_shuffle, d_obs: np.abs(d_shuffle) >= np.abs(d_obs)
    else:
        raise ValueError('tail must be "left", "right", or "both"')
    diff_obs = statistic(X[Y,:], axis=0) - statistic(X[~Y,:], axis=0)
    count = np.zeros(X.shape[1], dtype='float64')
    for i in range(numperm):
        Y_shuffle = np.random.permutation(Y)
        diff_shuffle = statistic(X[Y_shuffle,:], axis=0) - statistic(X[~Y_shuffle,:], axis=0)
        count += count_function(diff_shuffle, diff_obs)
    return diff_obs, (count + 1)/(numperm + 1) # upper bound pvalue
    
def univariate_vectorized_permtest(X, Y, numperm=10**5, tail='both'):
    # returns (difference between means pos and neg classes, pvalues), shape is num features x num binary classifications
    if Y.dtype != 'bool':
        Y = Y==1
    if tail == 'left': # how likely is it to see differences as small as observed
        count_function = lambda d_shuffle, d_obs: d_shuffle <= d_obs
    elif tail == 'right': # how likely is it to see differences as large as observed
        count_function = lambda d_shuffle, d_obs: d_shuffle >= d_obs
    elif tail == 'both': # how likely is it to see differences as extreme as observed
        count_function = lambda d_shuffle, d_obs: np.abs(d_shuffle) >= np.abs(d_obs)
    else:
        raise ValueError('tail must be "left", "right", or "both"')
    Xt = X.T
    positives = Y.sum(0).reshape(1,-1)
    negatives = (~Y).sum(0).reshape(1,-1)
    diff_obs = (Xt).dot(Y)/positives - (Xt).dot(~Y)/negatives
    count = np.zeros(diff_obs.shape, dtype='float64')
    for i in range(numperm):
        Y_shuffle = np.random.permutation(Y)
        diff_shuffle = (Xt).dot(Y_shuffle)/positives - (Xt).dot(~Y_shuffle)/negatives
        count += count_function(diff_shuffle, diff_obs)
    return diff_obs, (count + 1)/(numperm + 1) # upper bound pvalue

def univariate_grouppreserved_permtest(X, Y, G, numperm=10**5, statistic=np.mean, tail='both'):
    # returns (difference in statistic between pos and neg classes, pvalues)
    if Y.dtype != 'bool':
        Y = Y==1
    if tail == 'left': # how likely is it to see differences as small as observed
        count_function = lambda d_shuffle, d_obs: d_shuffle <= d_obs
    elif tail == 'right': # how likely is it to see differences as large as observed
        count_function = lambda d_shuffle, d_obs: d_shuffle >= d_obs
    elif tail == 'both': # how likely is it to see differences as extreme as observed
        count_function = lambda d_shuffle, d_obs: np.abs(d_shuffle) >= np.abs(d_obs)
    else:
        raise ValueError('tail must be "left", "right", or "both"')
    diff_obs = statistic(X[Y,:], axis=0) - statistic(X[~Y,:], axis=0)
    count = np.zeros(X.shape[1], dtype='float64')
    groups = np.unique(G)
    for i in range(numperm):
        Y_shuffle = np.zeros(Y.size, dtype='bool')
        for group in groups:
            Y_shuffle[G==group] = np.random.permutation(Y[G==group])
        diff_shuffle = statistic(X[Y_shuffle,:], axis=0) - statistic(X[~Y_shuffle,:], axis=0)
        count += count_function(diff_shuffle, diff_obs)
    return diff_obs, (count + 1)/(numperm + 1) # upper bound pvalue
     
def univariate_anova(X, Y):
    # returns (fvalues, pvalues)
    return feature_selection.f_classif(X, Y)
    
def univariate_utest(X, Y):
    # mann-whitney u-test
    # returns (uvalues, pvalues)
    if Y.dtype != 'bool':
        Y = Y==1
    u, p = np.apply_along_axis(lambda x,y: stats.mannwhitneyu(x[y], x[~y]), 0, X, Y)
    return u, 2*p
    
def nan_univariate_utest(X, Y):
    # mann-whitney u-test
    # returns (uvalues, pvalues)
    if Y.dtype != 'bool':
        Y = Y==1
    u, p = np.apply_along_axis(lambda x,y: stats.mannwhitneyu(x[np.logical_and(y,~np.isnan(x))], x[np.logical_and(~y,~np.isnan(x))]), 0, X, Y)
    return u, 2*p

def univariate_ttest(X, Y, equal_var=False):
    # Welch's t-test (equal_var=False) is preferred
    # sign of tvalue is determined by mean(positive_class) - mean(negative_class)
    # returns (tvalues, pvalues)
    if Y.dtype != 'bool':
        Y = Y==1
    return stats.ttest_ind(X[Y,:], X[~Y,:], axis=0, equal_var=equal_var, nan_policy='propagate')

def nan_univariate_ttest(X, Y, equal_var=False):
    # Welch's t-test (equal_var=False) is preferred
    # sign of tvalue is determined by mean(positive_class) - mean(negative_class)
    # returns (tvalues, pvalues)
    if Y.dtype != 'bool':
        Y = Y==1
    return stats.ttest_ind(X[Y,:], X[~Y,:], axis=0, equal_var=equal_var, nan_policy='omit')

def chisquare_table(x, y, xlevels=np.empty(0), ylevels=np.empty(0)):
    # returns (chi2value, pvalue, phicoef)
    if xlevels.size == 0:
        xlevels = np.unique(x)
    if ylevels.size == 0:
        ylevels = np.unique(y)
    contingency_table = np.zeros((ylevels.size, xlevels.size), dtype='int64')
    for i, ylevel in enumerate(ylevels):
        for j, xlevel in enumerate(xlevels):
            contingency_table[i,j] = (np.logical_and(y==ylevel, x==xlevel)).sum()
    chi2value, pvalue = stats.chi2_contingency(contingency_table)[:2]
    phicoef = np.sqrt(chi2value/contingency_table.sum())
    return chi2value, pvalue, phicoef
    
def nan_chisquare_table(x, y):
    # returns (chi2value, pvalue, phicoef)
    keep = ~np.isnan(x)
    y = y[keep]
    x = x[keep]
    x = x.astype('int8')
    y = y.astype('int8')
    xlevels = np.unique(x)
    ylevels = np.unique(y)
    if xlevels.size < 2 or ylevels.size < 2:
        return np.nan, np.nan, np.nan
    else:
        contingency_table = np.zeros((ylevels.size, xlevels.size), dtype='int64')
        for i, ylevel in enumerate(ylevels):
            for j, xlevel in enumerate(xlevels):
                contingency_table[i,j] = (np.logical_and(y==ylevel, x==xlevel)).sum()
        chi2value, pvalue = stats.chi2_contingency(contingency_table)[:2]
        phicoef = np.sqrt(chi2value/contingency_table.sum())
        return chi2value, pvalue, phicoef
    
def univariate_chisquare(X, Y):
    # chi-square test on R by C contingency table
    # R is number of classes and C is number of values of categorical variable X
    # returns (chi2values, pvalues, phicoefs)
    if Y.dtype != 'int8':
        Y = Y.astype('int8')
    if X.dtype != 'int8':
        X = X.astype('int8')
    Ylevels = np.unique(Y)
    return np.apply_along_axis(chisquare_table, 0, X, y=Y, ylevels=Ylevels)

def nan_univariate_chisquare(X, Y):
    # chi-square test on R by C contingency table
    # R is number of classes and C is number of values of categorical variable X
    # returns (chi2values, pvalues, phicoefs)
    return np.apply_along_axis(nan_chisquare_table, 0, X, y=Y)

def multiple_hypothesis_testing_correction(pvalues, alpha=0.05, method='fdr_bh'):
    is_ok = ~np.isnan(pvalues)
    is_significant = np.nan*np.zeros(pvalues.shape, dtype='float64')
    pvalues_corrected = np.nan*np.zeros(pvalues.shape, dtype='float64')
    if is_ok.any():
        is_significant[is_ok], pvalues_corrected[is_ok] = multicomp.multipletests(pvalues[is_ok], alpha=alpha, method=method)[:2]
    is_significant = is_significant==1
    return is_significant, pvalues_corrected

def class_difference(X, Y):
    if Y.dtype != 'bool':
        Y = Y==1
    positive_class_mean = (X[Y,:]).mean(axis=0)
    negative_class_mean = (X[~Y,:]).mean(axis=0)
    positive_class_median = np.median(X[Y,:], axis=0)
    negative_class_median = np.median(X[~Y,:], axis=0)
    class_mean_difference = positive_class_mean - negative_class_mean
    class_median_difference = positive_class_median - negative_class_median
    return class_mean_difference, class_median_difference
    
def nan_class_difference(X, Y):
    if Y.dtype != 'bool':
        Y = Y==1
    positive_class_mean = np.nanmean(X[Y,:], axis=0)
    negative_class_mean = np.nanmean(X[~Y,:], axis=0)
    positive_class_median = np.nanmedian(X[Y,:], axis=0)
    negative_class_median = np.nanmedian(X[~Y,:], axis=0)
    class_mean_difference = positive_class_mean - negative_class_mean
    class_median_difference = positive_class_median - negative_class_median
    return class_mean_difference, class_median_difference
