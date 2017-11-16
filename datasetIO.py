# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import os
import gzip
import pickle
import numpy as np
import dataclasses as dc

def load_datasetinfo(datasetspath):
    dataset_info = []
    with open(datasetspath, mode='rt', encoding="utf-8", errors="surrogateescape") as fr:
        fields = [x.strip() for x in fr.readline().split('\t')]
        for line in fr:
            entries = [x.strip() for x in line.split('\t')]
            dataset_info.append({field:entry for field,entry in zip(fields,entries)})
    return dataset_info

def save_datasetinfo(datasetspath, dataset_infos):
    fields = sorted(dataset_infos[0].keys())
    with open(datasetspath, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        fw.write('\t'.join(fields) + '\n')
        for dataset_info in dataset_infos:
            entries = [dataset_info[field] for field in fields]
            fw.write('\t'.join([entry if type(entry)==str else '{0:1.6g}'.format(entry) for entry in entries]) + '\n')

def append_datasetinfo(datasetspath, dataset_info):
    fields = sorted(dataset_info.keys())
    entries = [dataset_info[field] for field in fields]
    if not os.path.exists(datasetspath):
        with open(datasetspath, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
            fw.write('\t'.join(fields) + '\n')
    with open(datasetspath, mode='at', encoding='utf-8', errors='surrogateescape') as fw:
            fw.write('\t'.join([entry if type(entry)==str else '{0:1.6g}'.format(entry) for entry in entries]) + '\n')

def load_examples(examplespath):
    examples = set()
    with open(examplespath, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        fr.readline()
        for line in fr:
            examples.add(line.split('\t', maxsplit=1)[0].strip())
    return examples

def load_clusterassignments(clusterassignmentspath):
    if '.pickle' in clusterassignmentspath:
        with open(clusterassignmentspath, 'rb') as fr:
            return pickle.load(fr)
    else:
        item_cluster = {}
        with open(clusterassignmentspath, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
            fr.readline()
            for line in fr:
                item, cluster = [x.strip() for x in line.split('\t')]
                item_cluster[item] = int(cluster)
        return item_cluster

def save_clusterassignments(clusterassignmentspath, item_cluster, itemname):
    if '.pickle' in clusterassignmentspath:
        with open(clusterassignmentspath, 'wb') as fw:
            pickle.dump(item_cluster, fw)
    else:
        with open(clusterassignmentspath, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
            fw.write('\t'.join([itemname, 'cluster']) + '\n')
            for item, cluster in item_cluster.items():
                fw.write('\t'.join([item, str(cluster)]) + '\n')

def load_datamatrix(datasetpath, delimiter='\t', dtype='float64', getmetadata=True, getmatrix=True):
    if '.pickle' in datasetpath:
        with open(datasetpath, 'rb') as fr:
            return pickle.load(fr)
    else:
        if '.gz' in datasetpath:
            openfunc = gzip.open
        else:
            openfunc = open
        with openfunc(datasetpath, mode='rt', encoding="utf-8", errors="surrogateescape") as fr:
            rowmeta = {}
            columnmeta = {}
            rowlabels = []
            entries = [x.strip() for x in fr.readline().split(delimiter)]
            skipcolumns = sum([entry=='#' for entry in entries]) + 1
            columnname = entries[skipcolumns-1]
            columnlabels = np.array(entries[skipcolumns:], dtype='object')
            firstentry = entries[0]
            skiprows = 1
            if getmetadata:
                while firstentry == '#':
                    entries = [x.strip() for x in fr.readline().split(delimiter)]
                    columnmetaname = entries[skipcolumns-1].split('/')[-1]
                    if columnmetaname.lower() != 'na':
                        columnmeta[columnmetaname] = np.array(entries[skipcolumns:], dtype='object')
                    firstentry = entries[0]
                    skiprows += 1
                rowname = firstentry
                rowmetanames = entries[1:skipcolumns]
                if len(rowmetanames) > 0:
                    rowmetanames[-1] = rowmetanames[-1].split('/')[0]
                rowmetaname_idx = {}
                for i, rowmetaname in enumerate(rowmetanames):
                    if rowmetaname.lower() != 'na':
                        rowmeta[rowmetaname] = []
                        rowmetaname_idx[rowmetaname] = i
                for line in fr:
                    entries = [x.strip() for x in line.split(delimiter, maxsplit=skipcolumns)[:skipcolumns]]
                    rowlabels.append(entries.pop(0))
                    for rowmetaname, idx in rowmetaname_idx.items():
                        rowmeta[rowmetaname].append(entries[idx])
                rowlabels = np.array(rowlabels, dtype='object')
                for rowmetaname, rowmetavalues in rowmeta.items():
                    rowmeta[rowmetaname] = np.array(rowmetavalues, dtype='object')
            else:
                while firstentry == '#':
                    entries = [x.strip() for x in fr.readline().split(delimiter)]
                    firstentry = entries[0]
                    skiprows += 1
                rowname = firstentry
                for line in fr:
                    rowlabels.append(line.split(delimiter, maxsplit=1)[0].strip())
                rowlabels = np.array(rowlabels, dtype='object')
        if getmatrix:
            matrix = np.loadtxt(datasetpath, dtype=dtype, delimiter=delimiter, skiprows=skiprows,
                                usecols=range(skipcolumns,len(columnlabels)+skipcolumns), ndmin=2)
        else:
            matrix = np.zeros((0,0), dtype=dtype)
        matrixname = rowname + '_' + columnname + '_associations_from_' + datasetpath
        return dc.datamatrix(rowname, rowlabels, columnname, columnlabels, matrixname, matrix, rowmeta, columnmeta)

def save_datamatrix(datasetpath, dm):
    if '.pickle' in datasetpath:
        with open(datasetpath, 'wb') as fw:
            pickle.dump(dm, fw)
    else:
        if '.gz' in datasetpath:
            openfunc = gzip.open
        else:
            openfunc = open
        np.savetxt(datasetpath.replace('.txt', '.temp.txt'), dm.matrix, fmt='%1.6g', delimiter='\t', newline='\n')
        with openfunc(datasetpath, mode='wt', encoding="utf-8", errors="surrogateescape") as fw, openfunc(datasetpath.replace('.txt', '.temp.txt'), 'rt') as fr:
            rowmeta_names_and_dtypes = [(k,v.dtype) for k,v in dm.rowmeta.items()]
            spacers = ['#' for x in range(len(rowmeta_names_and_dtypes)+1)]
            fw.write('\t'.join(spacers + [dm.columnname] + dm.columnlabels.tolist()) + '\n')
            for columnmetaname, columnmetadata in dm.columnmeta.items():
                if columnmetadata.dtype == 'object':
                    fw.write('\t'.join(spacers + [columnmetaname] + columnmetadata.tolist()) + '\n')
                else:
                    fw.write('\t'.join(spacers + [columnmetaname] + ['{0:1.6g}'.format(x) for x in columnmetadata]) + '\n')
            fw.write('\t'.join([dm.rowname] + [k for k,t in rowmeta_names_and_dtypes] + ['na/na'] + ['na' for i in range(dm.shape[1])]) + '\n')
            for i, line in enumerate(fr):
                rowmetadata = [dm.rowmeta[k][i] if t=='object' else '{0:1.6g}'.format(dm.rowmeta[k][i]) for k,t in rowmeta_names_and_dtypes]
                fw.write('\t'.join([dm.rowlabels[i]] + rowmetadata + ['na']) + '\t' + line)
        os.remove(datasetpath.replace('.txt', '.temp.txt'))
