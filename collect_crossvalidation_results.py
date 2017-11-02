# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import sys
custompaths = ['/GWD/bioinfo/projects/cb01/users/rouillard/Python/Classes',
              '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Modules',
              '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Packages',
              '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Scripts']
# custompaths = ['C:\\Users\\ar988996\\Documents\\Python\\Classes',
#                'C:\\Users\\ar988996\\Documents\\Python\\Modules',
#                'C:\\Users\\ar988996\\Documents\\Python\\Packages',
#                'C:\\Users\\ar988996\\Documents\\Python\\Scripts']
for custompath in custompaths:
    if custompath not in sys.path:
        sys.path.append(custompath)
del custompath, custompaths

import os
import copy
import numpy as np
from machinelearning import datasetIO, dataclasses, modelevaluation

validation_reps = 200
validation_folds = 5

classifier_cutoff = 'mcc_cutoff'
classifier_stats = np.array(['p', 'n', 'ap', 'an', 'pp', 'pn', 'tp', 'fp', 'tn', 'fn', 'tpr', 'fpr', 'auroc', 'fnr', 'tnr',
                             'mcr', 'acc', 'fdr', 'ppv', 'auprc', 'fomr', 'npv', 'plr', 'nlr', 'dor', 'drr', 'darr',
                             'mrr', 'marr', 'f1s', 'mcc', 'fnlp'], dtype='object')

# classifier stats for each of 200 repetitions of cross-validation
stat_rep = dataclasses.datamatrix(rowname='classifier_performance_stat',
                            rowlabels=classifier_stats.copy(),
                            rowmeta={},
                            columnname='validation_rep',
                            columnlabels=np.array(['Rep'+str(x) for x in range(validation_reps)], dtype='object'),
                            columnmeta={'validation_folds':np.zeros(validation_reps, dtype='int64')},
                            matrixname='crossvalidation_classifier_performance_stats_across_validation_reps',
                            matrix=np.zeros((classifier_stats.size, validation_reps), dtype='float64'))

# classifier stats for each of 200reps*5folds=1000 train-test cycles
stat_fold = dataclasses.datamatrix(rowname='classifier_performance_stat',
                            rowlabels=classifier_stats.copy(),
                            rowmeta={},
                            columnname='validation_rep_and_fold',
                            columnlabels=np.full(validation_reps*validation_folds, '', dtype='object'),
                            columnmeta={'validation_rep':np.zeros(validation_reps*validation_folds, dtype='int64'),
                                        'validation_fold':np.zeros(validation_reps*validation_folds, dtype='int64'),
                                        'num_features':np.zeros(validation_reps*validation_folds, dtype='int64'),
                                        'features':np.full(validation_reps*validation_folds, '', dtype='object'),
                                        'model_type':np.full(validation_reps*validation_folds, '', dtype='object')},
                            matrixname='crossvalidation_classifier_performance_stats_across_validation_reps_and_folds',
                            matrix=np.zeros((classifier_stats.size, validation_reps*validation_folds), dtype='float64'))

# iterate over cross-validation reps and folds
print('iterating over cross-validation reps and folds...', flush=True)
R = 0
for validation_rep in range(validation_reps):
    Y = np.zeros(0, dtype='bool')
    P = np.zeros(0, dtype='float64')
    F = 0
    for validation_fold in range(validation_folds):
        gene_model_path = 'datasets/useful_features/rep{0!s}_fold{1!s}/gene_model_selected.txt.gz'.format(validation_rep, validation_fold)        
        stat_model_path = 'datasets/useful_features/rep{0!s}_fold{1!s}/stat_model_selected.txt.gz'.format(validation_rep, validation_fold)        
        if os.path.exists(gene_model_path):
            
            # load predictions for validation and unlabelled examples
            print('loading predictions for validation and unlabelled examples...', flush=True)
            gene_model = datasetIO.load_datamatrix(gene_model_path)
            stat_model = datasetIO.load_datamatrix(stat_model_path)
            isunknown = gene_model.rowmeta['class'] == 'unknown'
            Yf = gene_model.rowmeta['class'][~isunknown] == 'positive'
            Pf = gene_model.matrix[~isunknown,:].reshape(-1)
            
            # evaluate performance of predictions on individual fold
            print('evaluating performance of predictions on individual fold...', flush=True)
            stat_cut = modelevaluation.get_classifier_performance_stats(Y=Yf, P=Pf, classifier_stats=classifier_stats, plot_curves=False, get_priority_cutoffs=True)
            stat_fold.matrix[:,validation_rep*validation_folds+validation_fold] = stat_cut.matrix[:,stat_cut.columnmeta[classifier_cutoff]].reshape(-1)
            stat_fold.columnmeta['validation_rep'][validation_rep*validation_folds+validation_fold] = validation_rep
            stat_fold.columnmeta['validation_fold'][validation_rep*validation_folds+validation_fold] = validation_fold
            stat_fold.columnmeta['num_features'][validation_rep*validation_folds+validation_fold] = stat_model.columnmeta['num_features'][0]
            stat_fold.columnmeta['features'][validation_rep*validation_folds+validation_fold] = stat_model.columnmeta['features'][0]
            stat_fold.columnmeta['model_type'][validation_rep*validation_folds+validation_fold] = stat_model.columnmeta['model_type'][0]
            stat_fold.columnlabels[validation_rep*validation_folds+validation_fold] = 'Rep{0!s}Fold{1!s}'.format(validation_rep,validation_fold)
            print('    rep {0:1.3g} fold {1:1.3g} auroc {2:1.3g} auprc {3:1.3g}'.format(validation_rep, validation_fold, stat_fold.select('auroc',[])[validation_rep*validation_folds+validation_fold], stat_fold.select('auprc',[])[validation_rep*validation_folds+validation_fold]), flush=True)
            print('        model_type:{0} num_features:{1} features:{2}'.format(stat_model.columnmeta['model_type'][0], stat_model.columnmeta['num_features'][0], stat_model.columnmeta['features'][0]), flush=True)            
            print('        inner_loop auroc {0:1.3g} auprc {1:1.3g}'.format(stat_model.select('auroc_mean',[]), stat_model.select('auprc_mean',[])), flush=True)
            
            # collect fold predictions
            print('collecting fold predictions...', flush=True)            
            Y = np.append(Y, Yf)
            P = np.append(P, Pf)
            if F == 0:
                gene_fold = copy.deepcopy(gene_model)
            else:
                all_genes = np.union1d(gene_fold.rowlabels, gene_model.rowlabels)
                gene_fold = gene_fold.tolabels(rowlabels=all_genes, fillvalue=np.nan)
                gene_model = gene_model.tolabels(rowlabels=all_genes, fillvalue=np.nan)
                gene_fold.append(gene_model, 1)
            F += 1
            
    if F > 0:
        # evaluate performance of predictions on all folds
        print('evaluating performance of predictions on all folds...', flush=True)
        stat_cut = modelevaluation.get_classifier_performance_stats(Y=Y, P=P, classifier_stats=classifier_stats, plot_curves=False, get_priority_cutoffs=True)
        stat_rep.matrix[:,validation_rep] = stat_cut.matrix[:,stat_cut.columnmeta[classifier_cutoff]].reshape(-1)
        stat_rep.columnmeta['validation_folds'][validation_rep] = F
        gene_fold.matrix = np.nanmean(gene_fold.matrix, 1, keepdims=True)
        gene_fold.columnlabels = np.array(['Rep'+str(validation_rep)], dtype='object')
        gene_fold.columnmeta = {'validation_folds':np.array([F], dtype='int64')}
        gene_fold.columnname = 'validation_rep'
        gene_fold.updatesizeattribute()
        gene_fold.updateshapeattribute()
        if R == 0:
            gene_rep = copy.deepcopy(gene_fold)
        else:
            all_genes = np.union1d(gene_fold.rowlabels, gene_rep.rowlabels)
            gene_fold = gene_fold.tolabels(rowlabels=all_genes, fillvalue=np.nan)
            gene_rep = gene_rep.tolabels(rowlabels=all_genes, fillvalue=np.nan)
            gene_rep.append(gene_fold, 1)
        R += 1
        print('    rep {0:1.3g} folds {1:1.3g} auroc {2:1.3g} auprc {3:1.3g}'.format(validation_rep, F, stat_rep.select('auroc',[])[validation_rep], stat_rep.select('auprc',[])[validation_rep]), flush=True)

stat_fold.discard((stat_fold.matrix==0).all(0), 1)
stat_rep.discard((stat_rep.matrix==0).all(0), 1)

# save cross-validation performance stats for folds and reps
print('saving cross-validation performance stats for folds and reps...', flush=True)
datasetIO.save_datamatrix('datasets/useful_features/stat_fold_crossvalidation.pickle', stat_fold)
datasetIO.save_datamatrix('datasets/useful_features/stat_fold_crossvalidation.txt.gz', stat_fold)
datasetIO.save_datamatrix('datasets/useful_features/stat_rep_crossvalidation.pickle', stat_rep)
datasetIO.save_datamatrix('datasets/useful_features/stat_rep_crossvalidation.txt.gz', stat_rep)
datasetIO.save_datamatrix('datasets/useful_features/gene_rep_crossvalidation.pickle', gene_rep)
datasetIO.save_datamatrix('datasets/useful_features/gene_rep_crossvalidation.txt.gz', gene_rep)

print('done.', flush=True)
