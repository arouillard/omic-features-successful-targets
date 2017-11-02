# -*- coding: utf-8 -*-
"""
Performance Evaluation Module
"""


import numpy as np
import matplotlib.pyplot as plt
from machinelearning import dataclasses
from scipy import stats

def get_unique_pcuts(P, max_cuts=1000):
    uP = np.unique(P)[::-1]
    uP = np.insert(uP, 0, uP[0] + (uP[0]-uP[1])) if uP.size > 1 else np.insert(uP, 0, 1.01*uP[0])
    if uP.size > max_cuts:
        return uP[np.linspace(0, uP.size-1, max_cuts, dtype='int64')]
    else:
        return uP

def get_priority_cutoff_metadata(stat_cut, pp_min_frac=0.1, xx_min_frac=0.01):
    ap = stat_cut.select('ap',[])[0]
    pp_min = int(np.ceil(ap*pp_min_frac))
    xx_min = int(np.ceil(ap*xx_min_frac))
    is_qualified_cutoff = np.logical_and((stat_cut.matrix[np.in1d(stat_cut.rowlabels, ['tp', 'fp', 'tn', 'fn']),:] >= xx_min).all(0).reshape(-1),
                                         (stat_cut.matrix[stat_cut.rowlabels=='pp',:] >= pp_min).reshape(-1))
    if (~is_qualified_cutoff).all():
        is_qualified_cutoff[:] = True # if no qualified cutoffs, resort to all cutoffs
    mcc = stat_cut.select('mcc',[])
    ppv = stat_cut.select('ppv',[])
    mcc_max = mcc[is_qualified_cutoff].max()
    ppv_max = ppv[is_qualified_cutoff].max()
    mcc_idx = np.where(mcc >= mcc_max - 0.01*np.abs(mcc_max))[0][-1]
    ppv_idx = np.where(ppv >= ppv_max - 0.01*np.abs(ppv_max))[0][-1]
    p50_idx = np.argmin((stat_cut.select('p',[])-0.5)**2)
    ppe_idx = np.argmin((stat_cut.select('pp',[])-ap)**2)
    stat_cut.columnmeta['mcc_cutoff'] = np.arange(stat_cut.shape[1], dtype='int64') == mcc_idx
    stat_cut.columnmeta['ppv_cutoff'] = np.arange(stat_cut.shape[1], dtype='int64') == ppv_idx
    stat_cut.columnmeta['p50_cutoff'] = np.arange(stat_cut.shape[1], dtype='int64') == p50_idx
    stat_cut.columnmeta['ppe_cutoff'] = np.arange(stat_cut.shape[1], dtype='int64') == ppe_idx
    
def get_classifier_performance_stats(Y, P, uP=1000, classifier_stats='all', plot_curves=True, get_priority_cutoffs=True, pp_min_frac=0.1, xx_min_frac=0.01):
    if type(uP) == int:
        uP = get_unique_pcuts(P=P, max_cuts=uP).reshape(-1,1)
    elif len(uP.shape) == 1:
        uP = uP.reshape(-1,1)
    if type(classifier_stats) == str:
        classifier_stats = np.array(['p', 'n', 'ap', 'an', 'pp', 'pn', 'tp', 'fp', 'tn', 'fn', 'tpr', 'fpr', 'auroc', 'fnr', 'tnr',
                                     'mcr', 'acc', 'fdr', 'ppv', 'auprc', 'fomr', 'npv', 'plr', 'nlr', 'dor', 'drr', 'darr',
                                     'mrr', 'marr', 'f1s', 'mcc', 'fnlp'], dtype='object')
    n = np.float64(Y.size)  + 0.2
    ap = Y.sum().astype('float64')  + 0.1
    an = (~Y).sum().astype('float64')  + 0.1
    pp = (P >= uP).sum(1).astype('float64')  + 0.1
    pn = (P < uP).sum(1).astype('float64')  + 0.1
    tp = np.logical_and(P >= uP, Y).sum(1).astype('float64')  + 0.05 # if count is 5, then this introduces 1% error
    fp = np.logical_and(P >= uP, ~Y).sum(1).astype('float64')  + 0.05 # so don't take seriously any cut-off where
    tn = np.logical_and(P < uP, ~Y).sum(1).astype('float64')  + 0.05 # any count is less than 5
    fn = np.logical_and(P < uP, Y).sum(1).astype('float64')  + 0.05 # nnt is extremely sensitive to this adjustment, but not where nnt is actually reasonable
    uP = uP.reshape(-1)
    tpr = tp/ap # sensitivity, recall, 1-fnr
    fpr = fp/an # fall-out, 1-tnr, 1-specificity
    auroc = np.trapz(tpr, fpr)
    fnr = fn/ap # miss rate
    tnr = tn/an # specificity
    mcr = (fp + fn)/n
    acc = (tp + tn)/n
    fdr = fp/pp
    ppv = tp/pp # precision = 1-fdr
    auprc = np.trapz(ppv, tpr)
    fomr = fn/pn # false omission rate
    npv = tn/pn
    plr = (tp/fp)/(ap/an) # ratio of positives to negatives in positive predictions relative to ratio in whole sample, higher is better, tpr/fpr
    nlr = (fn/tn)/(ap/an) # ratio of positives to negatives in negative predictions relative to ratio in whole sample, lower is better, fnr/tnr
    dor = (tp/fp)/(fn/tn) # ratio of positives to negatives in positive predictions, divided by ratio of positives to negatives in negative predictions, positivelikelihoodratio/negativelikelihoodratio
    drr = (tp/pp)/(fn/pn) # relative risk or risk ratio, fraction of positives in positive predictions divided by fraction of positives in negative predictions, ppv/fomr
    darr = (tp/pp) - (fn/pn) # absolute risk reduction, fraction of positives in positive predictions minus fraction of positives in negative predictions, ppv - fomr
    mrr = (tp/pp)/(ap/n) # modified (by me) relative risk or risk ratio, fraction of positives in positive predictions divided by fraction of positives in whole sample, ppv/prevalence
    marr = (tp/pp) - (ap/n) # modified (by me) absolute risk reduction, fraction of positives in positive predictions minus fraction of positives in whole sample, ppv - prevalence
    f1s = 2*tp/(2*tp + fp + fn)
    mcc = (tp*tn - fp*fn)/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    fnlp = -stats.hypergeom.logsf(tp, n, ap, pp, loc=1)/np.log(10)
    results_dict = {'p':uP, 'n':n, 'ap':ap, 'an':an, 'pp':pp, 'pn':pn, 'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn, 'tpr':tpr,
                    'fpr':fpr, 'auroc':auroc, 'fnr':fnr, 'tnr':tnr, 'mcr':mcr, 'acc':acc, 'fdr':fdr, 'ppv':ppv,
                    'auprc':auprc, 'fomr':fomr, 'npv':npv, 'plr':plr, 'nlr':nlr, 'dor':dor, 'drr':drr, 'darr':darr,
                    'mrr':mrr, 'marr':marr, 'f1s':f1s, 'mcc':mcc, 'fnlp':fnlp}
    stat_cut = dataclasses.datamatrix(rowname='classifier_performance_stat', rowlabels=classifier_stats.copy(), rowmeta={},
                                      columnname='probability_cutoff', columnlabels=uP.copy(), columnmeta={},
                                      matrixname='classifier_performance_stats_vs_probability_cutoffs', matrix=np.zeros((classifier_stats.size, uP.size), dtype='float64'))
    for i, stat in enumerate(stat_cut.rowlabels):
        stat_cut.matrix[i,:] = results_dict[stat]
    if get_priority_cutoffs:
        get_priority_cutoff_metadata(stat_cut, pp_min_frac, xx_min_frac)
    if plot_curves:
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(fpr, tpr, 'k-')
        plt.ylabel('tpr, sensitivity, recall')
        plt.xlabel('fpr, 1-specificity, fall-out')
        plt.axis([0, 1, 0, 1])
        plt.subplot(2,2,2)
        plt.plot(tpr, ppv, 'k-')
        plt.ylabel('ppv, precision, 1-fdr')
        plt.xlabel('tpr, sensitivity, recall')
        plt.axis([0, 1, 0, 1])
        plt.subplot(2,2,3)
        plt.plot(uP, mcr, 'k-')
        plt.ylabel('mcr')
        plt.xlabel('p')
        plt.axis([0, 1, 0, 1])
        plt.gca().invert_xaxis()
        plt.subplot(2,2,4)
        plt.plot(uP, mcc, 'k-')
        plt.ylabel('mcc')
        plt.xlabel('p')
        plt.axis([0, 1, 0, 1])
        plt.gca().invert_xaxis()
    return stat_cut



def get_classifier_performance_stats(Y, P, uP=1000, classifier_stats='all', plot_curves=True, get_priority_cutoffs=True, pp_min_frac=0.1, xx_min_frac=0.01):
    if type(uP) == int:
        uP = get_unique_pcuts(P=P, max_cuts=uP).reshape(-1,1)
    elif len(uP.shape) == 1:
        uP = uP.reshape(-1,1)
    if type(classifier_stats) == str:
        classifier_stats = np.array(['p', 'n', 'ap', 'an', 'pp', 'pn', 'tp', 'fp', 'tn', 'fn', 'tpr', 'fpr', 'auroc', 'fnr', 'tnr',
                                     'mcr', 'acc', 'fdr', 'ppv', 'auprc', 'fomr', 'npv', 'plr', 'nlr', 'dor', 'drr', 'darr',
                                     'mrr', 'marr', 'f1s', 'mcc', 'fnlp'], dtype='object')
    n = np.float64(Y.size)  + 0.2
    ap = Y.sum().astype('float64')  + 0.1
    an = (~Y).sum().astype('float64')  + 0.1
    pp = (P >= uP).sum(1).astype('float64')  + 0.1
    pn = (P < uP).sum(1).astype('float64')  + 0.1
    tp = np.logical_and(P >= uP, Y).sum(1).astype('float64')  + 0.05 # if count is 5, then this introduces 1% error
    fp = np.logical_and(P >= uP, ~Y).sum(1).astype('float64')  + 0.05 # so don't take seriously any cut-off where
    tn = np.logical_and(P < uP, ~Y).sum(1).astype('float64')  + 0.05 # any count is less than 5
    fn = np.logical_and(P < uP, Y).sum(1).astype('float64')  + 0.05 # nnt is extremely sensitive to this adjustment, but not where nnt is actually reasonable
    uP = uP.reshape(-1)
    stat_fun_params = (uP, n, ap, an, pp, pn, tp, fp, tn, fn)
    stat_fun = {'p':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:P,
                'n':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:N,
                'ap':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:AP,
                'an':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:AN,
                'pp':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:PP,
                'pn':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:PN,
                'tp':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:TP,
                'fp':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:FP,
                'tn':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:TN,
                'fn':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:FN,
                'tpr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:TP/AP,
                'fpr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:FP/AN,
                'auroc':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:np.trapz(TP/AP,FP/AN),
                'fnr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:FN/AP,
                'tnr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:TN/AN,
                'mcr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:(FP + FN)/N,
                'acc':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:(TP + TN)/N,
                'fdr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:FP/PP,
                'ppv':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:TP/PP,
                'auprc':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:np.trapz(TP/PP,TP/AP),
                'fomr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:FN/PN,
                'npv':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:TN/PN,
                'plr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:(TP/FP)/(AP/AN),
                'nlr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:(FN/TN)/(AP/AN),
                'dor':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:(TP/FP)/(FN/TN),
                'drr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:(TP/PP)/(FN/PN),
                'darr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:(TP/PP) - (FN/PN),
                'mrr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:(TP/PP)/(AP/N),
                'marr':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:(TP/PP) - (AP/N),
                'f1s':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:2*TP/(2*TP + FP + FN),
                'mcc':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:(TP*TN - FP*FN)/np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)),
                'fnlp':lambda P, N, AP, AN, PP, PN, TP, FP, TN, FN:-stats.hypergeom.logsf(TP, N, AP, PP, loc=1)/np.log(10)}
    stat_cut = dataclasses.datamatrix(rowname='classifier_performance_stat', rowlabels=classifier_stats.copy(), rowmeta={},
                                      columnname='probability_cutoff', columnlabels=uP.copy(), columnmeta={},
                                      matrixname='classifier_performance_stats_vs_probability_cutoffs', matrix=np.zeros((classifier_stats.size, uP.size), dtype='float64'))
    for i, stat in enumerate(stat_cut.rowlabels):
        stat_cut.matrix[i,:] = stat_fun[stat](*stat_fun_params)
    if get_priority_cutoffs:
        get_priority_cutoff_metadata(stat_cut, pp_min_frac, xx_min_frac)
    if plot_curves:
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(stat_cut.select('fpr',[]), stat_cut.select('tpr',[]), 'k-')
        plt.ylabel('tpr, sensitivity, recall')
        plt.xlabel('fpr, 1-specificity, fall-out')
        plt.axis([0, 1, 0, 1])
        plt.subplot(2,2,2)
        plt.plot(stat_cut.select('tpr',[]), stat_cut.select('ppv',[]), 'k-')
        plt.ylabel('ppv, precision, 1-fdr')
        plt.xlabel('tpr, sensitivity, recall')
        plt.axis([0, 1, 0, 1])
        plt.subplot(2,2,3)
        plt.plot(stat_cut.select('p',[]), stat_cut.select('mcr',[]), 'k-')
        plt.ylabel('mcr')
        plt.xlabel('p')
        plt.axis([0, 1, 0, 1])
        plt.gca().invert_xaxis()
        plt.subplot(2,2,4)
        plt.plot(stat_cut.select('p',[]), stat_cut.select('mcc',[]), 'k-')
        plt.ylabel('mcc')
        plt.xlabel('p')
        plt.axis([0, 1, 0, 1])
        plt.gca().invert_xaxis()
    return stat_cut



