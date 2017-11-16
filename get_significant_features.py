# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import os
import numpy as np
import datasetIO
import featureselection

def main():
    
    # load dataset info
    print('loading dataset info...', flush=True)
    dataset_info_path = 'datasets/nonredundant_features/dataset_info.txt'
    dataset_infos = datasetIO.load_datasetinfo(dataset_info_path)
    
    # specify results folder
    print('specifying results folder...', flush=True)
    results_folder = 'datasets/significant_features'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    
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
        
        # specify feature significance test parameters
        print('specifying feature significance test parameters...', flush=True)
        dataset_info['feature_significance_test_function'] = featureselection.univariate_permtest
        dataset_info['feature_significance_test_permutations'] = 100000
        dataset_info['multiple_hypothesis_testing_correction_function'] = featureselection.multiple_hypothesis_testing_correction
        dataset_info['multiple_hypothesis_testing_correction_method'] = 'fdr_by'
        dataset_info['multiple_hypothesis_testing_correction_threshold'] = 0.05
        print('   feature_significance_test_function: {0}'.format(dataset_info['feature_significance_test_function']), flush=True)
        print('   feature_significance_test_permutations: {0!s}'.format(dataset_info['feature_significance_test_permutations']), flush=True)
        print('   multiple_hypothesis_testing_correction_function: {0}'.format(dataset_info['multiple_hypothesis_testing_correction_function']), flush=True)
        print('   multiple_hypothesis_testing_correction_method: {0}'.format(dataset_info['multiple_hypothesis_testing_correction_method']), flush=True)
        print('   multiple_hypothesis_testing_correction_threshold: {0!s}'.format(dataset_info['multiple_hypothesis_testing_correction_threshold']), flush=True)
        
        
        # compute feature significance with multiple hypothesis testing correction
        print('computing feature significance with multiple hypothesis testing correction...', flush=True)
        isunknown = gene_atb.rowmeta['class'] == 'unknown'
        gene_atb.columnmeta['test_statistic_values'], gene_atb.columnmeta['pvalues'] = dataset_info['feature_significance_test_function'](X=gene_atb.matrix[~isunknown,:], Y=(gene_atb.rowmeta['class'][~isunknown] == 'positive'), numperm=dataset_info['feature_significance_test_permutations'])
        gene_atb.columnmeta['is_significant'], gene_atb.columnmeta['pvalues_corrected'] = dataset_info['multiple_hypothesis_testing_correction_function'](gene_atb.columnmeta['pvalues'], alpha=dataset_info['multiple_hypothesis_testing_correction_threshold'], method=dataset_info['multiple_hypothesis_testing_correction_method'])
        gene_atb.columnmeta['correlation_sign'] = np.sign(gene_atb.columnmeta['test_statistic_values'])
        if (gene_atb.columnmeta['pvalues'] < 1/dataset_info['feature_significance_test_permutations']).any():
            print('    warning: not enough permutations to establish all pvalues...', flush=True)
        tobediscarded = np.logical_or(np.isnan(gene_atb.columnmeta['pvalues']), np.isnan(gene_atb.columnmeta['pvalues_corrected']))
        if tobediscarded.any():
            gene_atb.discard(tobediscarded, axis=1)
        
        # prioritize features
        print('prioritizing features...', flush=True)
        sortedindices = np.argsort(gene_atb.columnmeta['pvalues_corrected'])
        gene_atb.reorder(sortedindices, axis=1)
        
        # save feature significance info
        print('saving feature significance info...', flush=True)
        with open('{0}/{1}_feature_significance_info.txt'.format(results_folder, dataset_info['abbreviation']), mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
            writelist = ['dataset',
                         'abbreviation',
                         'feature',
                         'test_statistic',
                         'pvalue',
                         'pvalue_corrected',
                         'is_significant',
                         'correlation_sign',
                         'preferred_rowstat',
                         'similar_features']
            fw.write('\t'.join(writelist) + '\n')
            for j, feature in enumerate(gene_atb.columnlabels):
                writelist = [dataset_info['name'],
                             dataset_info['abbreviation'],
                             feature,
                             '{0:1.5g}'.format(gene_atb.columnmeta['test_statistic_values'][j]),
                             '{0:1.5g}'.format(gene_atb.columnmeta['pvalues'][j]),
                             '{0:1.5g}'.format(gene_atb.columnmeta['pvalues_corrected'][j]),
                             '{0:1.5g}'.format(gene_atb.columnmeta['is_significant'][j]),
                             '{0:1.5g}'.format(gene_atb.columnmeta['correlation_sign'][j]),
                             gene_atb.columnmeta['preferred_rowstat'][j],
                             gene_atb.columnmeta['similar_features'][j]]
                fw.write('\t'.join(writelist) + '\n')
        
        # discard features that are not significant
        print('discarding features that are not significant...', flush=True)
        tobediscarded = ~gene_atb.columnmeta['is_significant']
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
            # save significant features
            print('    saving {0!s} significant features...'.format(gene_atb.shape[1]), flush=True)
            dataset_info['path'] = '{0}/{1}.txt.gz'.format(results_folder, dataset_info['abbreviation'])
            dataset_info['significant_genes'] = gene_atb.shape[0]
            dataset_info['significant_features'] = gene_atb.shape[1]
            dataset_info['feature_significance_test_function'] = 'featureselection.univariate_permtest'
            dataset_info['multiple_hypothesis_testing_correction_function'] = 'featureselection.multiple_hypothesis_testing_correction'
            datasetIO.save_datamatrix(dataset_info['path'], gene_atb)
            datasetIO.append_datasetinfo('{0}/dataset_info.txt'.format(results_folder), dataset_info)
            
    print('done.', flush=True)

if __name__ == '__main__':
    main()
