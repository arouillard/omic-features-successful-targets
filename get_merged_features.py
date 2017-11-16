# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import os
import copy
import numpy as np
from operator import itemgetter
import datasetIO

def main(validation_rep=0, validation_fold=0):
    
    # load dataset info
    print('loading dataset info...', flush=True)
    dataset_info_path = 'datasets/generalizable_features/rep{0!s}_fold{1!s}/dataset_info.txt'.format(validation_rep, validation_fold)
    dataset_infos = datasetIO.load_datasetinfo(dataset_info_path)
    
    # specify results folder
    print('specifying results folder...', flush=True)
    results_folder = 'datasets/merged_features/rep{0!s}_fold{1!s}'.format(validation_rep, validation_fold)
    results_folder_parts = results_folder.split('/')
    for i in range(len(results_folder_parts)):
        results_folder_part = '/'.join(results_folder_parts[:i+1])
        if not os.path.exists(results_folder_part):
            os.mkdir(results_folder_part)
    
    # exclude mouse and small datasets
    print('excluding mouse datasets and datasets with few genes...', flush=True)
    dataset_infos = [dataset_info for dataset_info in dataset_infos if 'mouse' not in dataset_info['abbreviation'] and int(dataset_info['generalizable_genes']) > 1900]
    
    # exclude brain atlas datasets unless they're the only choice
    not_brainatlas = ['brainatlas' not in dataset_info['abbreviation'] for dataset_info in dataset_infos]
    if sum(not_brainatlas) > 0:
        print('excluding brain atlas datasets...', flush=True)
        dataset_infos = [dataset_info for dataset_info,nba in zip(dataset_infos, not_brainatlas) if nba]
    
    # iterate over datasets
    print('iterating over datasets...', flush=True)
    for i, dataset_info in enumerate(dataset_infos):
            
        # load dataset
        print('loading dataset {0}...'.format(dataset_info['abbreviation']), flush=True)
        gene_atb_i = datasetIO.load_datamatrix(datasetpath=dataset_info['path'])
        gene_atb_i.columnmeta['generalizability_pvalues_corrected'] = gene_atb_i.columnmeta['generalizability_pvalues_corrected'].astype('float64')
        gene_atb_i.columnmeta['dataset_abbreviation'] = np.full(gene_atb_i.shape[1], dataset_info['abbreviation'], dtype='object')
        gene_atb_i.columnmeta['dataset_feature'] = gene_atb_i.columnlabels.copy()
        gene_atb_i.columnlabels += '_' + dataset_info['abbreviation']
        gene_atb_i.rowname = 'GeneSym'
        gene_atb_i.columnname = 'Feature'
        if dataset_info['abbreviation'] == 'gtextissue_cleaned':
            gene_atb_i.discard(gene_atb_i.rowlabels=='C12ORF55', 0) # pesky duplicate row
        print(gene_atb_i)

        # merge dataset
        print('merging dataset...', flush=True)
        if i==0:
            gene_atb = copy.deepcopy(gene_atb_i)
            gene_atb.matrixname = 'merged_target_features'
            print('    first dataset, no merge...', flush=True)
        else:
            common_genes = np.intersect1d(gene_atb.rowlabels, gene_atb_i.rowlabels)
            gene_atb = gene_atb.tolabels(rowlabels=common_genes)
            gene_atb_i = gene_atb_i.tolabels(rowlabels=common_genes)
            gene_atb.append(gene_atb_i, 1)
            print('    common_genes: {0!s}...'.format(common_genes.size), flush=True)
        
    # specify merged dataset info
    print('specifying merged dataset info...', flush=True)
    dataset_info = {'abbreviation':'merged',
                    'name':'Merged Generalizable Target Features',
                    'path':'{0}/{1}.txt.gz'.format(results_folder, 'merged'),
                    'feature_normalization':'min-max',
                    'feature_similarity_metric':'cosine',
                    'feature_similarity_threshold':np.sqrt(0.5),
                    'genes':gene_atb.shape[0],
                    'features':gene_atb.shape[1],
                    'positives':(gene_atb.rowmeta['class'] == 'positive').sum(),
                    'negatives':(gene_atb.rowmeta['class'] == 'negative').sum(),
                    'unknowns':(gene_atb.rowmeta['class'] == 'unknown').sum()}
    for field, entry in dataset_info.items():
        print('    {0}: {1!s}'.format(field, entry), flush=True)
        
    # normalize features
    print('normalizing features...', flush=True)
    gene_atb.columnmeta['min'] = gene_atb.matrix.min(0)
    gene_atb.columnmeta['max'] = gene_atb.matrix.max(0)
    gene_atb.matrix = (gene_atb.matrix - gene_atb.columnmeta['min'].reshape(1,-1))/(gene_atb.columnmeta['max'].reshape(1,-1) - gene_atb.columnmeta['min'].reshape(1,-1))
    
    # prioritize features
    print('prioritizing features by generalizability_pvalues_corrected...', flush=True)
    sortedindices = np.argsort(gene_atb.columnmeta['generalizability_pvalues_corrected'])
    gene_atb.reorder(sortedindices, axis=1)
    
    # calculate feature similarity
    print('calculating feature similarity...', flush=True)
    atb_atb = gene_atb.tosimilarity(axis=1, metric=dataset_info['feature_similarity_metric'])
    
    # prioritize feature groups
    print('prioritizing feature groups...', flush=True)
    are_similar_features = np.abs(atb_atb.matrix) > dataset_info['feature_similarity_threshold']
    feature_group_size = are_similar_features.sum(1).astype('float64')
    feature_group_score = (np.abs(atb_atb.matrix)*are_similar_features).sum(1)/feature_group_size
    feature_priority = np.zeros(gene_atb.shape[1], dtype='float64')
    feature_priority[gene_atb.columnmeta['dataset_feature']=='mean'] = 1.0
    feature_priority[gene_atb.columnmeta['dataset_feature']=='stdv'] = 0.5
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
                generalizability_pvalues_corrected = gene_atb.columnmeta['generalizability_pvalues_corrected'][similarity_hit]
                si = np.argsort(generalizability_pvalues_corrected)
                similar_features = similar_features[si]
                similarity_values = similarity_values[si]
                generalizability_pvalues_corrected = generalizability_pvalues_corrected[si]
                print('        similar_feature, similarity_value, generalizability_pvalue_corrected', flush=True)
                for similar_feature, similarity_value, generalizability_pvalue_corrected in zip(similar_features, similarity_values, generalizability_pvalues_corrected):
                    print('        {0}, {1:1.3g}, {2:1.3g}'.format(similar_feature, similarity_value, generalizability_pvalue_corrected), flush=True)
                # replace feature with best similar feature
                j = np.where(gene_atb.columnlabels == similar_features[0])[0][0]
                gene_atb.columnmeta['similar_features'][j] = '|'.join(similar_features.tolist())
                print('        replacing feature "{0}" with best similar feature "{1}"...'.format(feature, gene_atb.columnlabels[j]), flush=True)
                gene_atb.matrix[:,i] = gene_atb.matrix[:,j]
                gene_atb.columnlabels[i] = gene_atb.columnlabels[j]
                for field in gene_atb.columnmeta.keys():
                    gene_atb.columnmeta[field][i] = gene_atb.columnmeta[field][j]
                fw.write('\t'.join(['{0}|{1:1.6g}|{2:1.6g}'.format(f,s,p) for f,s,p in zip(similar_features, similarity_values, generalizability_pvalues_corrected)]) + '\n')
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
        # save merged nonredundant features
        print('    saving {0!s} merged nonredundant features...'.format(gene_atb.shape[1]), flush=True)
        dataset_info['nonredundant_genes'] = gene_atb.shape[0]
        dataset_info['nonredundant_features'] = gene_atb.shape[1]
        datasetIO.save_datamatrix(dataset_info['path'], gene_atb)
        datasetIO.append_datasetinfo('{0}/dataset_info.txt'.format(results_folder), dataset_info)
            
    print('done.', flush=True)        

if __name__ == '__main__':
    main()
