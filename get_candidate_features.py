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
import copy
from machinelearning import datasetIO, dataclasses

def main():
    
    # load class examples
    print('loading class examples...', flush=True)
    class_examples_folder = 'targets/pharmaprojects'
    class_examples = {'positive':datasetIO.load_examples('{0}/positive.txt'.format(class_examples_folder)),
                      'negative':datasetIO.load_examples('{0}/negative.txt'.format(class_examples_folder)),
                      'unknown':datasetIO.load_examples('{0}/unknown.txt'.format(class_examples_folder))}
    
    # load dataset info
    print('loading dataset info...', flush=True)
    dataset_info_path = 'datasets/harmonizome/dataset_info.txt'
    dataset_infos = datasetIO.load_datasetinfo(dataset_info_path)
    
    # specify results folder
    print('specifying results folder...', flush=True)
    results_folder = 'datasets/candidate_features'
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
        dataset_info['original_genes'] = gene_atb.shape[0]
        dataset_info['original_features'] = gene_atb.shape[1]
        
        # decide feature normalization
        print('deciding feature normalization...', flush=True)
        if ('standardized' in dataset_info['abbreviation'] or 'cleaned' in dataset_info['abbreviation']) and (gene_atb.matrix == 0).sum()/gene_atb.size <= 0.5:
            # dataset is many-valued and filled-in
            print('    dataset is many-valued and filled-in...', flush=True)
            print('    z-scoring features...', flush=True)
            dataset_info['feature_normalization'] = 'z-score'
            mnv = np.nanmean(gene_atb.matrix, axis=0, keepdims=True)
            sdv = np.nanstd(gene_atb.matrix, axis=0, keepdims=True)
            gene_atb.matrix = (gene_atb.matrix - mnv)/sdv
            gene_atb.columnmeta['mean'] = mnv.reshape(-1)
            gene_atb.columnmeta['stdv'] = sdv.reshape(-1)
        else:
            # dataset is binary or tertiary or sparse
            print('    dataset is binary, tertiary, or sparse...', flush=True)
            print('    no feature normalization...', flush=True)
            dataset_info['feature_normalization'] = 'none'
            
        # assign class labels to genes
        print('assigning class labels to genes...', flush=True)
        gene_atb.rowmeta['class'] = np.full(gene_atb.shape[0], 'unknown', dtype='object')
        gene_atb.rowmeta['class'][np.in1d(gene_atb.rowlabels, list(class_examples['positive']))] = 'positive'
        gene_atb.rowmeta['class'][np.in1d(gene_atb.rowlabels, list(class_examples['negative']))] = 'negative'
        
        # add dataset mean and stdv as features
        print('adding dataset mean and stdv as features...', flush=True)
        gene_stat = dataclasses.datamatrix(rowname=gene_atb.rowname,
                                           rowlabels=gene_atb.rowlabels.copy(),
                                           rowmeta=copy.deepcopy(gene_atb.rowmeta),
                                           columnname=gene_atb.columnname,
                                           columnlabels=np.array(['mean', 'stdv'], dtype='object'),
                                           columnmeta={},
                                           matrixname=gene_atb.matrixname,
                                           matrix=np.append(gene_atb.matrix.mean(1, keepdims=True), gene_atb.matrix.std(1, keepdims=True), 1))
        gene_atb.append(gene_stat, 1)
        gene_atb.columnmeta['isrowstat'] = np.in1d(gene_atb.columnlabels, gene_stat.columnlabels)
        del gene_stat
        
        # identify features with little information about labelled examples
        print('identifying features with little information about labelled examples...', flush=True)
        isunknown = gene_atb.rowmeta['class'] == 'unknown'
        tobediscarded = np.logical_or.reduce(((gene_atb.matrix[~isunknown,:] != 0).sum(axis=0) < 3, (gene_atb.matrix[~isunknown,:] != 1).sum(axis=0) < 3, np.isnan(gene_atb.matrix[~isunknown,:]).any(axis=0)))
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
            # save candidate features
            print('    saving {0!s} candidate features...'.format(gene_atb.shape[1]), flush=True)
            dataset_info['path'] = '{0}/{1}.txt.gz'.format(results_folder, dataset_info['abbreviation'])
            dataset_info['candidate_genes'] = gene_atb.shape[0]
            dataset_info['candidate_features'] = gene_atb.shape[1]
            dataset_info['positive_examples'] = (gene_atb.rowmeta['class'] == 'positive').sum()
            dataset_info['negative_examples'] = (gene_atb.rowmeta['class'] == 'negative').sum()
            dataset_info['unknown_examples'] = (gene_atb.rowmeta['class'] == 'unknown').sum()
            datasetIO.save_datamatrix(dataset_info['path'], gene_atb)
            datasetIO.append_datasetinfo('{0}/dataset_info.txt'.format(results_folder), dataset_info)
            
    print('done.', flush=True)

if __name__ == '__main__':
    main()
