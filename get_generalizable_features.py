# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import sys
#custompaths = ['/GWD/bioinfo/projects/cb01/users/rouillard/Python/Classes',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Modules',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Packages',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Scripts']
custompaths = ['C:\\Users\\ar988996\\Documents\\Python\\Classes',
               'C:\\Users\\ar988996\\Documents\\Python\\Modules',
               'C:\\Users\\ar988996\\Documents\\Python\\Packages',
               'C:\\Users\\ar988996\\Documents\\Python\\Scripts']
for custompath in custompaths:
    if custompath not in sys.path:
        sys.path.append(custompath)
del custompath, custompaths

import os
import numpy as np
from machinelearning import datasetIO, featureselection

def main(validation_rep=0, validation_fold=0):
    
    # load target clusters
    print('loading target cluster assignments...', flush=True)
    target_cluster_path = 'targets/clusters/gene_cluster_byfamily_revised2.pickle'
    gene_cluster = datasetIO.load_clusterassignments(target_cluster_path)
    
    # load dataset info
    print('loading dataset info...', flush=True)
    dataset_info_path = 'datasets/nonredundant_features/dataset_info.txt'
    dataset_infos = datasetIO.load_datasetinfo(dataset_info_path)
    
    # load validation examples
    print('loading validation examples...', flush=True)
    validation_examples_path = 'targets/validation_examples/rep{0!s}_fold{1!s}.txt'.format(validation_rep, validation_fold)
    with open(validation_examples_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        validation_examples = fr.read().split('\n')
    
    # specify results folder
    print('specifying results folder...', flush=True)
    results_folder = 'datasets/generalizable_features/rep{0!s}_fold{1!s}'.format(validation_rep, validation_fold)
    results_folder_parts = results_folder.split('/')
    for i in range(len(results_folder_parts)):
        results_folder_part = '/'.join(results_folder_parts[:i+1])
        if not os.path.exists(results_folder_part):
            os.mkdir(results_folder_part)
    
    # iterate over datasets
    print('iterating over datasets...', flush=True)
    for dataset_info in dataset_infos:
        
#        # just work with hpatissuesmrna for testing/debugging the pipeline
#        if dataset_info['abbreviation'] != 'hpatissuesmrna_cleaned':
#            print('skipping {0}. not in testing set...'.format(dataset_info['abbreviation']), flush=True)
#            continue
        
        # check if another python instance is already working on this dataset
        if os.path.exists('{0}/{1}_in_progress.txt'.format(results_folder, dataset_info['abbreviation'])):
            print('skipping {0}. already in progress...'.format(dataset_info['abbreviation']), flush=True)
            continue
        
        # log start of processing
        with open('{0}/{1}_in_progress.txt'.format(results_folder, dataset_info['abbreviation']), mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
            print('working on {0}...'.format(dataset_info['abbreviation']), flush=True)
            fw.write('working on {0}...'.format(dataset_info['abbreviation']))
            
        # load dataset
        print('loading dataset...', flush=True)
        gene_atb = datasetIO.load_datamatrix(datasetpath=dataset_info['path'])
        
        # specify feature generalizability test parameters
        print('specifying feature generalizability test parameters...', flush=True)
        dataset_info['feature_generalizability_test_function'] = featureselection.univariate_grouppreserved_permtest
        dataset_info['feature_generalizability_test_permutations'] = 10000 # 100000
        dataset_info['feature_generalizability_test_targetclusterpath'] = target_cluster_path
        dataset_info['multiple_hypothesis_testing_correction_function'] = featureselection.multiple_hypothesis_testing_correction
        dataset_info['multiple_hypothesis_testing_correction_method'] = 'fdr_by'
        dataset_info['multiple_hypothesis_testing_correction_threshold'] = 0.05
        print('   feature_generalizability_test_function: {0}'.format(dataset_info['feature_generalizability_test_function']), flush=True)
        print('   feature_generalizability_test_permutations: {0!s}'.format(dataset_info['feature_generalizability_test_permutations']), flush=True)
        print('   feature_generalizability_test_targetclusterpath: {0}'.format(dataset_info['feature_generalizability_test_targetclusterpath']), flush=True)
        print('   multiple_hypothesis_testing_correction_function: {0}'.format(dataset_info['multiple_hypothesis_testing_correction_function']), flush=True)
        print('   multiple_hypothesis_testing_correction_method: {0}'.format(dataset_info['multiple_hypothesis_testing_correction_method']), flush=True)
        print('   multiple_hypothesis_testing_correction_threshold: {0!s}'.format(dataset_info['multiple_hypothesis_testing_correction_threshold']), flush=True)
        
        # exclude validation and unlabeled examples from significance calculation
        print('excluding validation and unlabeled examples from significance calculation...', flush=True)
        isvalidation = np.in1d(gene_atb.rowlabels, validation_examples)
        isunknown = gene_atb.rowmeta['class'] == 'unknown'
        istraintest = ~np.logical_or(isvalidation, isunknown)
        
        # compute feature generalizability with multiple hypothesis testing correction
        print('computing feature generalizability with multiple hypothesis testing correction...', flush=True)
        gene_atb.rowmeta['cluster'] = np.array([gene_cluster[g] if g in gene_cluster else -1 for g in gene_atb.rowlabels], dtype='int64')
        gene_atb.columnmeta['generalizability_test_statistic_values'], gene_atb.columnmeta['generalizability_pvalues'] = dataset_info['feature_generalizability_test_function'](X=gene_atb.matrix[istraintest,:], Y=(gene_atb.rowmeta['class'][istraintest] == 'positive'), G=gene_atb.rowmeta['cluster'][istraintest], numperm=dataset_info['feature_generalizability_test_permutations'])
        gene_atb.columnmeta['is_generalizable'], gene_atb.columnmeta['generalizability_pvalues_corrected'] = dataset_info['multiple_hypothesis_testing_correction_function'](gene_atb.columnmeta['generalizability_pvalues'], alpha=dataset_info['multiple_hypothesis_testing_correction_threshold'], method=dataset_info['multiple_hypothesis_testing_correction_method'])
        gene_atb.columnmeta['generalizability_correlation_sign'] = np.sign(gene_atb.columnmeta['generalizability_test_statistic_values'])
        if (gene_atb.columnmeta['generalizability_pvalues'] < 1/dataset_info['feature_generalizability_test_permutations']).any():
            print('    warning: not enough permutations to establish all pvalues...', flush=True)
        tobediscarded = np.logical_or(np.isnan(gene_atb.columnmeta['generalizability_pvalues']), np.isnan(gene_atb.columnmeta['generalizability_pvalues_corrected']))
        if tobediscarded.any():
            gene_atb.discard(tobediscarded, axis=1)
        
        # prioritize features
        print('prioritizing features...', flush=True)
        sortedindices = np.argsort(gene_atb.columnmeta['generalizability_pvalues_corrected'])
        gene_atb.reorder(sortedindices, axis=1)
        
        # save feature generalizability info
        print('saving feature generalizability info...', flush=True)
        with open('{0}/{1}_feature_generalizability_info.txt'.format(results_folder, dataset_info['abbreviation']), mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
            writelist = ['dataset',
                         'abbreviation',
                         'feature',
                         'generalizability_test_statistic',
                         'generalizability_pvalue',
                         'generalizability_pvalue_corrected',
                         'is_generalizable',
                         'generalizability_correlation_sign',
                         'preferred_rowstat',
                         'similar_features']
            fw.write('\t'.join(writelist) + '\n')
            for j, feature in enumerate(gene_atb.columnlabels):
                writelist = [dataset_info['name'],
                             dataset_info['abbreviation'],
                             feature,
                             '{0:1.5g}'.format(gene_atb.columnmeta['generalizability_test_statistic_values'][j]),
                             '{0:1.5g}'.format(gene_atb.columnmeta['generalizability_pvalues'][j]),
                             '{0:1.5g}'.format(gene_atb.columnmeta['generalizability_pvalues_corrected'][j]),
                             '{0:1.5g}'.format(gene_atb.columnmeta['is_generalizable'][j]),
                             '{0:1.5g}'.format(gene_atb.columnmeta['generalizability_correlation_sign'][j]),
                             gene_atb.columnmeta['preferred_rowstat'][j],
                             gene_atb.columnmeta['similar_features'][j]]
                fw.write('\t'.join(writelist) + '\n')
        
        # discard features that are not generalizable
        print('discarding features that are not generalizable...', flush=True)
        tobediscarded = ~gene_atb.columnmeta['is_generalizable']
        if tobediscarded.any():
            # discard features
            print('    discarding {0!s} features. {1!s} features remaining...'.format(tobediscarded.sum(), (~tobediscarded).sum()), flush=True)
            gene_atb.discard(tobediscarded, axis=1)
        else:
            # keep all features
            print('    no features to discard. {0!s} features remaining...'.format(gene_atb.shape[1]), flush=True)
        
        # save if dataset has content
        print('saving if dataset has content...', flush=True)
        if gene_atb.shape[0] == 0 or gene_atb.shape[1] == 0:
            # no content
            print('    nothing to save...', flush=True)
        else:
            # save generalizable features
            print('    saving {0!s} generalizable features...'.format(gene_atb.shape[1]), flush=True)
            dataset_info['path'] = '{0}/{1}.txt.gz'.format(results_folder, dataset_info['abbreviation'])
            dataset_info['generalizable_genes'] = gene_atb.shape[0]
            dataset_info['generalizable_features'] = gene_atb.shape[1]
            dataset_info['feature_generalizability_test_function'] = 'featureselection.univariate_grouppreserved_permtest'
            dataset_info['multiple_hypothesis_testing_correction_function'] = 'featureselection.multiple_hypothesis_testing_correction'
            datasetIO.save_datamatrix(dataset_info['path'], gene_atb)
            datasetIO.append_datasetinfo('{0}/dataset_info.txt'.format(results_folder), dataset_info)
            
    print('done.', flush=True)

if __name__ == '__main__':
    main()
