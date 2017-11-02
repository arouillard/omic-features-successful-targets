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
from machinelearning import datasetIO
from operator import itemgetter

def main():
    
    # load dataset info
    print('loading dataset info...', flush=True)
    dataset_info_path = 'datasets/candidate_features/dataset_info.txt'
    dataset_infos = datasetIO.load_datasetinfo(dataset_info_path)
    
    # specify results folder
    print('specifying results folder...', flush=True)
    results_folder = 'datasets/nonredundant_features'
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
        gene_atb.columnmeta['isrowstat'] = gene_atb.columnmeta['isrowstat'].astype('int64').astype('bool')
        
        # decide feature similarity metric
        print('deciding feature similarity metric...', flush=True)
        if ('standardized' in dataset_info['abbreviation'] or 'cleaned' in dataset_info['abbreviation']) and (gene_atb.matrix == 0).sum()/gene_atb.size <= 0.5:
            # dataset is many-valued and filled-in
            print('    dataset is many-valued and filled-in...', flush=True)
            print('    using spearman for similarity...', flush=True)
            dataset_info['feature_similarity_metric'] = 'spearman'
            dataset_info['feature_similarity_threshold'] = np.sqrt(0.5)
        else:
            # dataset is binary or tertiary or sparse
            print('    dataset is binary, tertiary, or sparse...', flush=True)
            print('    using cosine for similarity...', flush=True)
            dataset_info['feature_similarity_metric'] = 'cosine'
            dataset_info['feature_similarity_threshold'] = np.sqrt(0.5)
            
        # calculate feature similarity
        print('calculating feature similarity...', flush=True)
        atb_atb = gene_atb.tosimilarity(axis=1, metric=dataset_info['feature_similarity_metric'])
        
        # prioritize feature groups
        print('prioritizing feature groups...', flush=True)
        are_similar_features = np.abs(atb_atb.matrix) > dataset_info['feature_similarity_threshold']
        feature_group_size = are_similar_features.sum(1).astype('float64')
        feature_group_score = (np.abs(atb_atb.matrix)*are_similar_features).sum(1)/feature_group_size
        feature_priority = np.zeros(gene_atb.shape[1], dtype='float64')
        feature_priority[gene_atb.columnlabels=='mean'] = 1.0
        feature_priority[gene_atb.columnlabels=='stdv'] = 0.5
        feature_infos = list(zip(np.arange(gene_atb.shape[1], dtype='int64'), gene_atb.columnlabels.copy(), feature_group_size.copy(), feature_priority.copy(), feature_group_score.copy()))
        feature_infos.sort(key=itemgetter(4), reverse=True)
        feature_infos.sort(key=itemgetter(3), reverse=True)
        feature_infos.sort(key=itemgetter(2), reverse=True)
    #        for feature_info in feature_infos:
    #            print('{0:1.3g}, {1}, {2:1.3g}, {3:1.3g}, {4:1.3g}'.format(feature_info[0], feature_info[1], feature_info[2], feature_info[3], feature_info[4]))
        sorted_feature_indices = np.array([feature_info[0] for feature_info in feature_infos], dtype='int64')
        atb_atb.reorder(sorted_feature_indices, axis=0)
        atb_atb.reorder(sorted_feature_indices, axis=1)
        gene_atb.reorder(sorted_feature_indices, axis=1)
        are_similar_features = are_similar_features[sorted_feature_indices,:][:,sorted_feature_indices]
        
        # group similar features
        print('grouping similar features...', flush=True)
        tobediscarded = np.zeros(gene_atb.shape[1], dtype='bool')
        gene_atb.columnmeta['similar_features'] = np.full(gene_atb.shape[1], '', dtype='object')
        gene_atb.columnmeta['preferred_rowstat'] = np.full(gene_atb.shape[1], '', dtype='object')
        rowstats = gene_atb.columnlabels[gene_atb.columnmeta['isrowstat']]
        with open('{0}/{1}_feature_groups.txt'.format(results_folder, dataset_info['abbreviation']), mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
            for i, feature in enumerate(gene_atb.columnlabels):
                if ~tobediscarded[i]:
                    # find similar features
                    print('    finding features similar to feature "{0}"...'.format(feature), flush=True)
                    similarity_hit = are_similar_features[i,:]
                    similarity_hit = np.logical_and(similarity_hit, ~tobediscarded) # just what's new
                    similarity_hit[:i] = False
                    similar_features = gene_atb.columnlabels[similarity_hit]
                    similarity_values = atb_atb.matrix[i,similarity_hit]
                    rowstat_is_in_group = np.in1d(rowstats, similar_features)
                    gene_atb.columnmeta['similar_features'][i] = '|'.join(similar_features.tolist())
                    if rowstat_is_in_group.any():
                        # replace feature with summary stat
                        gene_atb.columnmeta['preferred_rowstat'][i] = rowstats[rowstat_is_in_group.nonzero()[0][0]]
                        gene_atb.matrix[:,i] = gene_atb.select([],gene_atb.columnmeta['preferred_rowstat'][i])
                        print('        replacing feature "{0}" with summary stat "{1}"...'.format(feature, gene_atb.columnmeta['preferred_rowstat'][i]), flush=True)
                    elif similarity_hit.sum() > 1:
                        # replace feature with group average
                        print('        replacing feature "{0}" with average of {1!s} features...'.format(feature, similarity_hit.sum()), flush=True)
                        feature_weight = atb_atb.matrix[i,similarity_hit]
                        feature_weight = feature_weight/np.sum(np.abs(feature_weight))
                        gene_atb.matrix[:,i] = (gene_atb.matrix[:,similarity_hit]*(feature_weight.reshape(1,-1))).sum(1)
                    else:
                        print('        no similar features...', flush=True)
                    fw.write('\t'.join(['{0}|{1:1.6g}'.format(f,v) for f,v in zip(similar_features, similarity_values)]) + '\n')
                    similarity_hit[i] = False
                    tobediscarded = np.logical_or(tobediscarded, similarity_hit)
    
        # discard features absorbed into group features
        print('discarding features absorbed into group features...', flush=True)
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
            # save nonredundant features
            print('    saving {0!s} nonredundant features...'.format(gene_atb.shape[1]), flush=True)
            dataset_info['path'] = '{0}/{1}.txt.gz'.format(results_folder, dataset_info['abbreviation'])
            dataset_info['nonredundant_genes'] = gene_atb.shape[0]
            dataset_info['nonredundant_features'] = gene_atb.shape[1]
            datasetIO.save_datamatrix(dataset_info['path'], gene_atb)
            datasetIO.append_datasetinfo('{0}/dataset_info.txt'.format(results_folder), dataset_info)
            
    print('done.', flush=True)

if __name__ == '__main__':
    main()
