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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def main(validation_rep=0, validation_fold=0):

    # load dataset info
    print('loading dataset info...', flush=True)
    dataset_info_path = 'datasets/merged_features/rep{0!s}_fold{1!s}/dataset_info.txt'.format(validation_rep, validation_fold)
    dataset_info = datasetIO.load_datasetinfo(dataset_info_path)[0]
    
    # load validation examples
    print('loading validation examples...', flush=True)
    validation_examples_path = 'targets/validation_examples/rep{0!s}_fold{1!s}.txt'.format(validation_rep, validation_fold)
    with open(validation_examples_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        validation_examples = fr.read().split('\n')
    
    # specify results folder
    print('specifying results folder...', flush=True)
    results_folder = 'datasets/useful_features/rep{0!s}_fold{1!s}'.format(validation_rep, validation_fold)
    results_folder_parts = results_folder.split('/')
    for i in range(len(results_folder_parts)):
        results_folder_part = '/'.join(results_folder_parts[:i+1])
        if not os.path.exists(results_folder_part):
            os.mkdir(results_folder_part)
    
    # load dataset
    print('loading dataset {0}...'.format(dataset_info['abbreviation']), flush=True)
    gene_atb = datasetIO.load_datamatrix(datasetpath=dataset_info['path'])
    
    # specify cross-validation parameters
    print('specifying cross-validation parameters...', flush=True)
    reps = 20
    folds = 5
    rf_trees = 1000
    include_logistic_regression = True
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    print('    reps: {0!s}'.format(reps))
    print('    folds: {0!s}'.format(folds))
    
    # initialize models
    print('initializing models...', flush=True)
    rfmodel = RandomForestClassifier(n_estimators=rf_trees, oob_score=False, n_jobs=-1, class_weight='balanced')
    print(rfmodel)
    lrmodel = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1e3, fit_intercept=True, intercept_scaling=1e3, class_weight='balanced', random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    print(lrmodel)
    
    # initialize data matrices for collecting model feature importances and cross-validation performance stats
    print('initializing data matrices for collecting model feature importances and cross-validation performance stats...', flush=True)
    classifier_stats = np.array(['p', 'n', 'ap', 'an', 'pp', 'pn', 'tp', 'fp', 'tn', 'fn', 'tpr', 'fpr', 'auroc', 'fnr', 'tnr',
                                 'mcr', 'acc', 'fdr', 'ppv', 'auprc', 'fomr', 'npv', 'plr', 'nlr', 'dor', 'drr', 'darr',
                                 'mrr', 'marr', 'f1s', 'mcc', 'fnlp'], dtype='object')
    sm = dataclasses.datamatrix(rowname='classifier_performance_stat',
                                rowlabels=classifier_stats.copy(),
                                rowmeta={},
                                columnname='model',
                                columnlabels=np.array(['M'+str(x) for x in range(gene_atb.shape[1])], dtype='object'),
                                columnmeta={'num_features':np.zeros(gene_atb.shape[1], dtype='int64'), 'features':np.full(gene_atb.shape[1], '', dtype='object'), 'oob_score':np.zeros(gene_atb.shape[1], dtype='float64')},
                                matrixname='crossvalidation_classifier_performance_stats_vs_models',
                                matrix=np.zeros((classifier_stats.size, gene_atb.shape[1]), dtype='float64'))
    stat_model_rf_mean = copy.deepcopy(sm)
    stat_model_rf_stdv = copy.deepcopy(sm)
    stat_model_lr_mean = copy.deepcopy(sm)
    stat_model_lr_stdv = copy.deepcopy(sm)
    del sm
    fm = dataclasses.datamatrix(rowname=gene_atb.columnname,
                                rowlabels=gene_atb.columnlabels.copy(),
                                rowmeta=copy.deepcopy(gene_atb.columnmeta),
                                columnname='model',
                                columnlabels=np.array(['M'+str(x) for x in range(gene_atb.shape[1])], dtype='object'),
                                columnmeta={'num_features':np.zeros(gene_atb.shape[1], dtype='int64'), 'features':np.full(gene_atb.shape[1], '', dtype='object'), 'oob_score':np.zeros(gene_atb.shape[1], dtype='float64')},
                                matrixname='model_feature_importances',
                                matrix=np.zeros((gene_atb.shape[1], gene_atb.shape[1]), dtype='float64'))
    feature_model_rf = copy.deepcopy(fm)
    feature_model_lr = copy.deepcopy(fm)
    del fm
    
    # exclude validation and unlabeled examples from cross-validation loop
    print('excluding validation and unlabeled examples from cross-validation loop...', flush=True)
    isvalidation = np.in1d(gene_atb.rowlabels, validation_examples)
    isunknown = gene_atb.rowmeta['class'] == 'unknown'
    istraintest = ~np.logical_or(isvalidation, isunknown)
    Y = (gene_atb.rowmeta['class'][istraintest] == 'positive')
    #X = gene_atb.matrix[istraintest,:]
    
    # perform incremental feature elimination with cross-validation
    print('performing incremental feature elimination with cross-validation...', flush=True)
    for i in range(gene_atb.shape[1]):
        print('    features: {0!s}...'.format(gene_atb.shape[1]-i), flush=True)
        if i == 0:
            hit_rf = np.ones(gene_atb.shape[1], dtype='bool')
            hit_lr = np.ones(gene_atb.shape[1], dtype='bool')
        else:
            hit_rf = feature_model_rf.matrix[:,i-1] > feature_model_rf.matrix[feature_model_rf.matrix[:,i-1] > 0,i-1].min()
            #hit_lr = feature_model_lr.matrix[:,i-1] > feature_model_lr.matrix[feature_model_lr.matrix[:,i-1] > 0,i-1].min()
            hit_lr = hit_rf
        X_rf = gene_atb.matrix[istraintest,:][:,hit_rf]
        X_lr = gene_atb.matrix[istraintest,:][:,hit_lr]
        stat_rep_rf = np.zeros((classifier_stats.size, reps), dtype='float64')
        stat_rep_lr = np.zeros((classifier_stats.size, reps), dtype='float64')
        fi_rep_rf = np.zeros((X_rf.shape[1], reps), dtype='float64')
        fi_rep_lr = np.zeros((X_lr.shape[1], reps), dtype='float64')
        for rep in range(reps):
            print('        rep {0!s} of {1!s}...'.format(rep+1, reps), flush=True)
            Ptest_rf = np.zeros(Y.size, dtype='float64')
            Ptest_lr = np.zeros(Y.size, dtype='float64')
            fi_fold_rf = np.zeros((X_rf.shape[1], folds), dtype='float64')
            fi_fold_lr = np.zeros((X_lr.shape[1], folds), dtype='float64')
            for fold, (train_indices, test_indices) in enumerate(skf.split(X_rf, Y)):
                print('            fold {0!s} of {1!s}...'.format(fold+1, folds), flush=True)
                Y_train = Y[train_indices]
                X_rf_train = X_rf[train_indices]
                X_lr_train = X_lr[train_indices]
                #Y_test = Y[test_indices]
                X_rf_test = X_rf[test_indices]
                X_lr_test = X_lr[test_indices]
                rfmodel.fit(X_rf_train, Y_train)
                Ptest_rf[test_indices] = rfmodel.predict_proba(X_rf_test)[:,rfmodel.classes_==1].reshape(-1)
                fi_fold_rf[:,fold] = rfmodel.feature_importances_
                lrmodel.fit(X_lr_train, Y_train)
                Ptest_lr[test_indices] = lrmodel.predict_proba(X_lr_test)[:,lrmodel.classes_==1].reshape(-1)
                fi_fold_lr[:,fold] = np.abs(lrmodel.coef_.reshape(-1))
            fi_rep_rf[:,rep] = fi_fold_rf.mean(1)
            stat_cut = modelevaluation.get_classifier_performance_stats(Y=Y, P=Ptest_rf, classifier_stats=classifier_stats, plot_curves=False, get_priority_cutoffs=True)
            stat_rep_rf[:,rep] = stat_cut.matrix[:,stat_cut.columnmeta['p50_cutoff']].reshape(-1)
            fi_rep_lr[:,rep] = fi_fold_lr.mean(1)
            stat_cut = modelevaluation.get_classifier_performance_stats(Y=Y, P=Ptest_lr, classifier_stats=classifier_stats, plot_curves=False, get_priority_cutoffs=True)
            stat_rep_lr[:,rep] = stat_cut.matrix[:,stat_cut.columnmeta['p50_cutoff']].reshape(-1)
        feature_model_rf.matrix[hit_rf,i] = fi_rep_rf.mean(1)
        feature_model_rf.columnmeta['num_features'][i] = gene_atb.shape[1]-i
        feature_model_rf.columnmeta['features'][i] = '|'.join(gene_atb.columnlabels[hit_rf].tolist())
        stat_model_rf_mean.matrix[:,i] = stat_rep_rf.mean(1)
        stat_model_rf_mean.columnmeta['num_features'][i] = gene_atb.shape[1]-i
        stat_model_rf_mean.columnmeta['features'][i] = '|'.join(gene_atb.columnlabels[hit_rf].tolist())
        stat_model_rf_stdv.matrix[:,i] = stat_rep_rf.std(1)
        stat_model_rf_stdv.columnmeta['num_features'][i] = gene_atb.shape[1]-i
        stat_model_rf_stdv.columnmeta['features'][i] = '|'.join(gene_atb.columnlabels[hit_rf].tolist())
        feature_model_lr.matrix[hit_lr,i] = fi_rep_lr.mean(1)
        feature_model_lr.columnmeta['num_features'][i] = gene_atb.shape[1]-i
        feature_model_lr.columnmeta['features'][i] = '|'.join(gene_atb.columnlabels[hit_lr].tolist())
        stat_model_lr_mean.matrix[:,i] = stat_rep_lr.mean(1)
        stat_model_lr_mean.columnmeta['num_features'][i] = gene_atb.shape[1]-i
        stat_model_lr_mean.columnmeta['features'][i] = '|'.join(gene_atb.columnlabels[hit_lr].tolist())
        stat_model_lr_stdv.matrix[:,i] = stat_rep_lr.std(1)
        stat_model_lr_stdv.columnmeta['num_features'][i] = gene_atb.shape[1]-i
        stat_model_lr_stdv.columnmeta['features'][i] = '|'.join(gene_atb.columnlabels[hit_lr].tolist())
    
    # concatenate data matrices with model feature importances
    print('concatenating data matrices with model feature importances...', flush=True)
    feature_model_rf.columnlabels += '_rf'
    feature_model_rf.columnmeta['model_type'] = np.full(feature_model_rf.shape[1], 'random_forest', dtype='object')
    feature_model_lr.columnlabels += '_lr'
    feature_model_lr.columnmeta['model_type'] = np.full(feature_model_lr.shape[1], 'logistic_regression', dtype='object')
    feature_model_rf.append(feature_model_lr, 1)
    feature_model = feature_model_rf
    del feature_model_rf, feature_model_lr
    
    # concatenate data matrices with model cross-validation performance stats
    print('concatenating data matrices with model cross-validation performance stats...', flush=True)
    stat_model_rf_mean.rowlabels += '_mean'
    stat_model_rf_stdv.rowlabels += '_stdv'
    stat_model_rf_mean.append(stat_model_rf_stdv, 0)
    stat_model_rf_mean.columnlabels += '_rf'
    stat_model_rf_mean.columnmeta['model_type'] = np.full(stat_model_rf_mean.shape[1], 'random_forest', dtype='object')
    stat_model_lr_mean.rowlabels += '_mean'
    stat_model_lr_stdv.rowlabels += '_stdv'
    stat_model_lr_mean.append(stat_model_lr_stdv, 0)
    stat_model_lr_mean.columnlabels += '_lr'
    stat_model_lr_mean.columnmeta['model_type'] = np.full(stat_model_lr_mean.shape[1], 'logistic_regression', dtype='object')
    stat_model_rf_mean.append(stat_model_lr_mean, 1)
    stat_model = stat_model_rf_mean
    del stat_model_rf_mean
    
    # select simplest model (fewest features) with auroc and auprc within 95% of max
    print('selecting simplest model (fewest features) with auroc and auprc within 95% of max...', flush=True)
    model_scores = 0.5*(stat_model.select('auroc_mean',[]) + stat_model.select('auprc_mean',[]))
    if include_logistic_regression:
        selected_model_index = np.where(model_scores >= 0.95*model_scores.max())[0][-1]
    else:
        selected_model_index = np.where(np.logical_and(model_scores >= 0.95*model_scores[stat_model.columnmeta['model_type']=='random_forest'].max(), stat_model.columnmeta['model_type']=='random_forest'))[0][-1]
    selected_model_name = stat_model.columnlabels[selected_model_index]
    selected_model_features = feature_model.rowlabels[feature_model.matrix[:,selected_model_index] != 0]
    selected_model_type = stat_model.columnmeta['model_type'][selected_model_index]
    selected_model = rfmodel if selected_model_type=='random_forest' else lrmodel
    gene_atb = gene_atb.tolabels(columnlabels=selected_model_features)
    feature_model_selected = feature_model.tolabels(columnlabels=selected_model_name)
    stat_model_selected = stat_model.tolabels(columnlabels=selected_model_name)
    print('    selected_model_name: {0}'.format(selected_model_name), flush=True)
    print('    selected_model_features: {0}'.format('|'.join(selected_model_features)), flush=True)
    
    
    
    
    
    # iterate over selected features to rebuild design matrix
    print('iterating over selected features to rebuild design matrix...', flush=True)
    for i, (selected_feature, dataset_abbreviation) in enumerate(zip(gene_atb.columnlabels, gene_atb.columnmeta['dataset_abbreviation'])):
        
        # load dataset
        print('    loading dataset {0}...'.format(dataset_abbreviation), flush=True)
        dataset_path = 'datasets/generalizable_features/rep{0!s}_fold{1!s}/{2}.txt.gz'.format(validation_rep, validation_fold, dataset_abbreviation)
        gene_atb_i = datasetIO.load_datamatrix(dataset_path)
        gene_atb_i.columnmeta['generalizability_pvalues_corrected'] = gene_atb_i.columnmeta['generalizability_pvalues_corrected'].astype('float64')
        gene_atb_i.columnmeta['dataset_abbreviation'] = np.full(gene_atb_i.shape[1], dataset_abbreviation, dtype='object')
        gene_atb_i.columnmeta['dataset_feature'] = gene_atb_i.columnlabels.copy()
        gene_atb_i.columnlabels += '_' + dataset_abbreviation
        gene_atb_i.rowname = 'GeneSym'
        gene_atb_i.columnname = 'Feature'
        if dataset_abbreviation == 'gtextissue_cleaned':
            gene_atb_i.discard(gene_atb_i.rowlabels=='C12ORF55', 0) # pesky duplicate row
        print(gene_atb_i)
        
        # select feature
        print('    selecting feature {0}...'.format(selected_feature), flush=True)
        gene_atb_i.discard(gene_atb_i.columnlabels != selected_feature, 1)

        # merge dataset
        print('    merging dataset...', flush=True)
        if i==0:
            gene_atb_selected = copy.deepcopy(gene_atb_i)
            gene_atb_selected.matrixname = 'merged_target_features'
            print('        first dataset, no merge...', flush=True)
        else:
            common_genes = np.intersect1d(gene_atb_selected.rowlabels, gene_atb_i.rowlabels)
            gene_atb_selected = gene_atb_selected.tolabels(rowlabels=common_genes)
            gene_atb_i = gene_atb_i.tolabels(rowlabels=common_genes)
            gene_atb_selected.append(gene_atb_i, 1)
            print('        common_genes: {0!s}...'.format(common_genes.size), flush=True)
        
    # normalize features
    print('normalizing features...', flush=True)
    gene_atb_selected.columnmeta['min'] = gene_atb_selected.matrix.min(0)
    gene_atb_selected.columnmeta['max'] = gene_atb_selected.matrix.max(0)
    gene_atb_selected.matrix = (gene_atb_selected.matrix - gene_atb_selected.columnmeta['min'].reshape(1,-1))/(gene_atb_selected.columnmeta['max'].reshape(1,-1) - gene_atb_selected.columnmeta['min'].reshape(1,-1))
        
    # update metadata
    print('updating metadata...', flush=True)
    assert (gene_atb.columnlabels == gene_atb_selected.columnlabels).all()
    for field, values in gene_atb.columnmeta.items():
        if field not in gene_atb_selected.columnmeta:
            gene_atb_selected.columnmeta[field] = values
    print('old_num_genes:{0!s}\tnew_num_genes:{1!s}'.format(gene_atb.shape[0], gene_atb_selected.shape[0]), flush=True)
    del gene_atb
    
    
    
    
    
    
    # refit selected model
    print('refitting selected model...', flush=True)
    isvalidation = np.in1d(gene_atb_selected.rowlabels, validation_examples)
    isunknown = gene_atb_selected.rowmeta['class'] == 'unknown'
    istraintest = ~np.logical_or(isvalidation, isunknown)
    selected_model.fit(gene_atb_selected.matrix[istraintest,:], gene_atb_selected.rowmeta['class'][istraintest] == 'positive')
    
    # get predictions for validation and unlabelled examples
    print('getting predictions for validation and unlabelled examples...', flush=True)
    gene_model_selected = dataclasses.datamatrix(rowname=gene_atb_selected.rowname,
                                rowlabels=gene_atb_selected.rowlabels.copy(),
                                rowmeta=copy.deepcopy(gene_atb_selected.rowmeta),
                                columnname=stat_model_selected.columnname,
                                columnlabels=stat_model_selected.columnlabels.copy(),
                                columnmeta=copy.deepcopy(stat_model_selected.columnmeta),
                                matrixname='success_probabilities_for_validation_and_unlabelled_examples',
                                matrix=selected_model.predict_proba(gene_atb_selected.matrix)[:,selected_model.classes_==1])
    gene_model_selected.discard(istraintest, 0)
    
    # save results
    print('saving {0!s} useful features and model results...'.format(gene_atb_selected.shape[1]), flush=True)
    dataset_info['path'] = '{0}/{1}.txt.gz'.format(results_folder, dataset_info['abbreviation'])
    dataset_info['selected_model_name'] = selected_model_name
    dataset_info['selected_model_features'] = '|'.join(selected_model_features)
    dataset_info['selected_model_type'] = selected_model_type
    dataset_info['crossvalidation_reps'] = reps
    dataset_info['crossvalidation_folds'] = folds
    dataset_info['rf_trees'] = rf_trees
    dataset_info['include_logistic_regression'] = include_logistic_regression
    for stat_name, stat_values in zip(stat_model_selected.rowlabels, stat_model_selected.matrix):
        dataset_info[stat_name] = stat_values.item()
    datasetIO.save_datamatrix(dataset_info['path'], gene_atb_selected)
    datasetIO.save_datamatrix('{0}/stat_model.txt.gz'.format(results_folder), stat_model)
    datasetIO.save_datamatrix('{0}/feature_model.txt.gz'.format(results_folder), feature_model)
    datasetIO.save_datamatrix('{0}/stat_model_selected.txt.gz'.format(results_folder), stat_model_selected)
    datasetIO.save_datamatrix('{0}/feature_model_selected.txt.gz'.format(results_folder), feature_model_selected)
    datasetIO.save_datamatrix('{0}/gene_model_selected.txt.gz'.format(results_folder), gene_model_selected)
    datasetIO.append_datasetinfo('{0}/dataset_info.txt'.format(results_folder), dataset_info)
            
    print('done.', flush=True)

if __name__ == '__main__':
    main()
