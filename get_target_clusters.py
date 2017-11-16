# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import pickle
import numpy as np
import datasetIO
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
import fastcluster
from sklearn.metrics import silhouette_score

def main():
    
    # load gene family membership from HGNC
    print('loading gene family membership from HGNC...', flush=True)
    with open('targets/clusters/target_family_matrix.pickle', 'rb') as fr:
        gf = pickle.load(fr)
    gf.matrix = (gf.matrix != 0).astype('int64')
    
    # load class examples
    print('loading class examples...', flush=True)
    class_examples_folder = 'targets/pharmaprojects'
    class_examples = {'positive':datasetIO.load_examples('{0}/positive.txt'.format(class_examples_folder)),
                      'negative':datasetIO.load_examples('{0}/negative.txt'.format(class_examples_folder)),
                      'unknown':datasetIO.load_examples('{0}/unknown.txt'.format(class_examples_folder))}
    
    # filter to targets with phase III outcomes
    print('filtering to targets with phase III outcomes...', flush=True)
    gf = gf.tolabels(rowlabels=list(class_examples['positive'].union(class_examples['negative'])))
    tobediscarded = np.logical_or((gf.matrix == 0).all(0), gf.columnlabels=='Other')
    gf.discard(tobediscarded, 1)
    tobediscarded = (gf.matrix == 0).all(1)
    gf.discard(tobediscarded, 0)
    
    # cluster targets according to membership in HGNC families
    print('clustering targets according to membership in HGNC families...', flush=True)
    D = distance.pdist(gf.matrix, 'cosine')
    Z = fastcluster.linkage(D, 'average')
    D = distance.squareform(D)
    numclusters = np.arange(2, int(gf.shape[0]/1)-1, 1, dtype='int64')
    silhouette = np.zeros_like(numclusters, dtype='float64')
    for i, nc in enumerate(numclusters):
        C = hierarchy.cut_tree(Z, nc).reshape(-1)
        silhouette[i] = silhouette_score(D, C, 'precomputed')
    plt.figure(); plt.plot(numclusters, silhouette, '-k')
    selectednumclusters = numclusters[silhouette == silhouette[~np.isnan(silhouette)].max()][0]
    gf.rowmeta['cluster'] = hierarchy.cut_tree(Z, selectednumclusters).reshape(-1)
    gf.rowmeta['clustered_order'] = hierarchy.leaves_list(Z).astype('int64')
    
    # eliminate single target clusters
    print('eliminating single target clusters...', flush=True)
    count = np.zeros(selectednumclusters, dtype='int64')
    for i in range(selectednumclusters):
        count[i] = (gf.rowmeta['cluster'] == i).sum()
    plt.figure(); plt.hist(count)
    minclustersize = 2
    smallclusters = (count < minclustersize).nonzero()[0]
    gf.rowmeta['cluster'][np.in1d(gf.rowmeta['cluster'], smallclusters)] = selectednumclusters
    for i, c in enumerate(np.sort(np.unique(gf.rowmeta['cluster']))):
        gf.rowmeta['cluster'][gf.rowmeta['cluster']==c] = i
        
    # vizualize clustergram
    print('visualizing clustergram...', flush=True)
    gf.cluster(1)
    gf.reorder(gf.rowmeta['clustered_order'].copy(), 0)
    gf.heatmap(['cluster'],[])
    
    # create dictionary assigning targets to clusters
    print('creating dictionary assigning targets to clusters...', flush=True)
    gene_cluster = {g:i for g in class_examples['positive'].union(class_examples['negative'])}
    gene_cluster.update({g:c for g,c in zip(gf.rowlabels, gf.rowmeta['cluster'])})
    
    # save assignments of targets to clusters by membership in HGNC families
    print('saving assignments of targets to clusters by membership in HGNC families...', flush=True)
    with open('targets/clusters/gene_cluster_byfamily.pickle', 'wb') as fw:
        pickle.dump(gene_cluster, fw)
            
    print('done.', flush=True)

if __name__ == '__main__':
    main()
