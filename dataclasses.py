# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import copy
import numpy as np
import stats as mlstats
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
import fastcluster
import matplotlib.pyplot as plt

class datamatrix(object):
    """A datamatrix object stores a dataset in matrix format and defines functions
    for operating on the data in the datamatrix format. Rows and columns have
    labels and an arbitrary number of additional metadata.
    
    Attributes:
        rowlabels: numpyarray containing row labels
        columnlabels: numpyarray containing column labels
        rowname: string defining what the row labels are
        columnname: string defining what the column labels are
        rowmeta: dictionary containing fieldname:numpyarray pairs
        columnmeta: dictionary containing fieldname:numpyarray pairs
        matrix: numpyarray containing values for row,column pairs
        matrixname: string defining what the matrix values are
        size: numpyscalar reporting total number of elements in matrix
        shape: tuple reporting (number of rows, number of columns) in matrix
        
        
    Methods:
        updatesizeattribute: update size attribute to current size
        updateshapeattribute: update shape attribute to current shape
        reorder: sort rows or columns of the matrix as well as the corresponding
            labels and metadata
        discard: remove rows or columns of the matrix as well as the corresponding
            labels and metadata
    """
    
    def __init__(self, rowname, rowlabels, columnname, columnlabels, matrixname, matrix, rowmeta={}, columnmeta={}):
        """Create a datamatrix object."""
        self.rowname = rowname
        self.rowlabels = rowlabels
        self.rowmeta = rowmeta
        self.columnname = columnname
        self.columnlabels = columnlabels
        self.columnmeta = columnmeta
        self.matrixname = matrixname
        self.matrix = matrix
        self.updatesizeattribute()
        self.updateshapeattribute()
        self.updatedtypeattribute()
        
    def __str__(self):
        description = 'Data matrix: Rowname={0}, Columnname={1}, Matrixname={2}, Shape=({3!s}, {4!s}), Dtype={5}'\
                        .format(self.rowname, self.columnname, self.matrixname, self.shape[0], self.shape[1], self.dtype)
        return description

    def updatesizeattribute(self):
        self.size = self.matrix.size

    def updateshapeattribute(self):
        self.shape = self.matrix.shape
        
    def updatedtypeattribute(self):
        self.dtype = self.matrix.dtype

    def reorder(self, sortedindices, axis):
        if type(sortedindices[0]) != np.int64 and type(sortedindices[0]) != np.int32 and type(sortedindices[0]) != int:
            raise TypeError('sortedindices must be numpy.int64 or int')
        if axis==0:
            if sortedindices.size != self.shape[0]:
                raise ValueError('size of sortedindices array does not match number of rows')
            self.matrix = self.matrix[sortedindices,:]
            self.rowlabels = self.rowlabels[sortedindices]
            for field, values in self.rowmeta.items():
                self.rowmeta[field] = values[sortedindices]
        elif axis==1:
            if sortedindices.size != self.shape[1]:
                raise ValueError('size of sortedindices array does not match number of columns')
            self.matrix = self.matrix[:,sortedindices]
            self.columnlabels = self.columnlabels[sortedindices]
            for field, values in self.columnmeta.items():
                self.columnmeta[field] = values[sortedindices]
        else:
            raise ValueError('invalid axis')

    def discard(self, tobediscarded, axis):
        if type(tobediscarded[0]) != np.bool_ and type(tobediscarded[0]) != bool:
            raise TypeError('tobediscarded must be numpy.bool_ or bool')
        elif tobediscarded.sum() == 0:
            print('Nothing to be discarded. Data matrix unchanged.')
        else:
            if axis==0:
                if tobediscarded.size != self.shape[0]:
                    raise ValueError('size of tobediscarded array does not match number of rows')
                else:
                    self.matrix = self.matrix[~tobediscarded,:]
                    self.rowlabels = self.rowlabels[~tobediscarded]
                    for field, values in self.rowmeta.items():
                        self.rowmeta[field] = values[~tobediscarded]
            elif axis==1:
                if tobediscarded.size != self.shape[1]:
                    raise ValueError('size of tobediscarded array does not match number of columns')
                else:
                    self.matrix = self.matrix[:,~tobediscarded]
                    self.columnlabels = self.columnlabels[~tobediscarded]
                    for field, values in self.columnmeta.items():
                        self.columnmeta[field] = values[~tobediscarded]
            else:
                raise ValueError('invalid axis')
            self.updatesizeattribute()
            self.updateshapeattribute()
    
    def vertcat(self, other, orderby):
        if self.shape[1] != other.shape[1] or len(set(self.columnlabels).intersection(other.columnlabels)) != len(set(self.columnlabels).union(other.columnlabels)):
            raise ValueError('columns do not overlap')
        else:
            if orderby == 'self':
                other = other.tolabels(columnlabels=self.columnlabels)
            elif orderby == 'other':
                self = self.tolabels(columnlabels=other.columnlabels)
            else:
                raise ValueError('orderby must be "self" or "other"')
            catted = datamatrix(rowname=self.rowname,
                                rowlabels=np.concatenate((self.rowlabels, other.rowlabels)),
                                columnname=self.columnname,
                                columnlabels=self.columnlabels.copy(),
                                matrixname=self.matrixname,
                                matrix=np.concatenate((self.matrix, other.matrix), axis=0),
                                rowmeta={},
                                columnmeta=copy.deepcopy(self.columnmeta))
            for field, values in other.columnmeta.items():
                if field not in catted.columnmeta:
                    catted.columnmeta[field] = values.copy()
            for field, values in self.rowmeta.items():
                if field in other.rowmeta:
                    catted.rowmeta[field] = np.concatenate((values, other.rowmeta[field]))
                else:
                    if values.dtype == 'object':
                        catted.rowmeta[field] = np.concatenate((values, np.full(other.shape[0], '', dtype='object')))
                    else:
                        catted.rowmeta[field] = np.concatenate((values, np.zeros(other.shape[0], dtype=values.dtype)))
            for field, values in other.rowmeta.items():
                if field not in catted.rowmeta:
                    if values.dtype == 'object':
                        catted.rowmeta[field] = np.concatenate((np.full(self.shape[0], '', dtype='object'), values))
                    else:
                        catted.rowmeta[field] = np.concatenate((np.zeros(self.shape[0], dtype=values.dtype), values))
            catted.updatesizeattribute()
            catted.updateshapeattribute()
            catted.updatedtypeattribute()
            return catted
            
    def horzcat(self, other, orderby):
        if self.shape[0] != other.shape[0] or len(set(self.rowlabels).intersection(other.rowlabels)) != len(set(self.rowlabels).union(other.rowlabels)):
            raise ValueError('rows do not overlap')
        else:
            if orderby == 'self':
                other = other.tolabels(rowlabels=self.rowlabels)
            elif orderby == 'other':
                self = self.tolabels(rowlabels=other.rowlabels)
            else:
                raise ValueError('orderby must be "self" or "other"')
            catted = datamatrix(rowname=self.rowname,
                                rowlabels=self.rowlabels.copy(),
                                columnname=self.columnname,
                                columnlabels=np.concatenate((self.columnlabels, other.columnlabels)),
                                matrixname=self.matrixname,
                                matrix=np.concatenate((self.matrix, other.matrix), axis=1),
                                rowmeta=copy.deepcopy(self.rowmeta),
                                columnmeta={})
            for field, values in other.rowmeta.items():
                if field not in catted.rowmeta:
                    catted.rowmeta[field] = values.copy()
            for field, values in self.columnmeta.items():
                if field in other.columnmeta:
                    catted.columnmeta[field] = np.concatenate((values, other.columnmeta[field]))
                else:
                    if values.dtype == 'object':
                        catted.columnmeta[field] = np.concatenate((values, np.full(other.shape[1], '', dtype='object')))
                    else:
                        catted.columnmeta[field] = np.concatenate((values, np.zeros(other.shape[1], dtype=values.dtype)))
            for field, values in other.columnmeta.items():
                if field not in catted.columnmeta:
                    if values.dtype == 'object':
                        catted.columnmeta[field] = np.concatenate((np.full(self.shape[1], '', dtype='object'), values))
                    else:
                        catted.columnmeta[field] = np.concatenate((np.zeros(self.shape[1], dtype=values.dtype), values))
            catted.updatesizeattribute()
            catted.updateshapeattribute()
            catted.updatedtypeattribute()
            return catted
    
    def concatenate(self, other, orderby, axis):
        if axis == 0:
            return self.vertcat(other, orderby)
        elif axis == 1:
            return self.horzcat(other, orderby)
        else:
            raise ValueError('invalid axis')
            
    def vertapp(self, other):
        if self.shape[1] != other.shape[1] or len(set(self.columnlabels).intersection(other.columnlabels)) != len(set(self.columnlabels).union(other.columnlabels)):
            raise ValueError('columns do not overlap')
        else:
            other = other.tolabels(columnlabels=self.columnlabels)
            self.rowlabels = np.concatenate((self.rowlabels, other.rowlabels))
            self.matrix = np.concatenate((self.matrix, other.matrix), axis=0)
            for field, values in other.columnmeta.items():
                if field not in self.columnmeta:
                    self.columnmeta[field] = values.copy()
                else:
                    if values.dtype == 'object':
                        is_missing = self.columnmeta[field] == ''
                    else:
                        is_missing = self.columnmeta[field] == 0
                    if is_missing.any():
                        self.columnmeta[field][is_missing] = values[is_missing]
            for field, values in self.rowmeta.items():
                if field in other.rowmeta:
                    self.rowmeta[field] = np.concatenate((values, other.rowmeta[field]))
                else:
                    if values.dtype == 'object':
                        self.rowmeta[field] = np.concatenate((values, np.full(other.shape[0], '', dtype='object')))
                    else:
                        self.rowmeta[field] = np.concatenate((values, np.zeros(other.shape[0], dtype=values.dtype)))
            for field, values in other.rowmeta.items():
                if field not in self.rowmeta:
                    if values.dtype == 'object':
                        self.rowmeta[field] = np.concatenate((np.full(self.shape[0], '', dtype='object'), values))
                    else:
                        self.rowmeta[field] = np.concatenate((np.zeros(self.shape[0], dtype=values.dtype), values))
            self.updatesizeattribute()
            self.updateshapeattribute()
            self.updatedtypeattribute()
            
    def horzapp(self, other):
        if self.shape[0] != other.shape[0] or len(set(self.rowlabels).intersection(other.rowlabels)) != len(set(self.rowlabels).union(other.rowlabels)):
            raise ValueError('rows do not overlap')
        else:
            other = other.tolabels(rowlabels=self.rowlabels)
            self.columnlabels=np.concatenate((self.columnlabels, other.columnlabels))
            self.matrix=np.concatenate((self.matrix, other.matrix), axis=1)
            for field, values in other.rowmeta.items():
                if field not in self.rowmeta:
                    self.rowmeta[field] = values.copy()
                else:
                    if values.dtype == 'object':
                        is_missing = self.rowmeta[field] == ''
                    else:
                        is_missing = self.rowmeta[field] == 0
                    if is_missing.any():
                        self.rowmeta[field][is_missing] = values[is_missing]
            for field, values in self.columnmeta.items():
                if field in other.columnmeta:
                    self.columnmeta[field] = np.concatenate((values, other.columnmeta[field]))
                else:
                    if values.dtype == 'object':
                        self.columnmeta[field] = np.concatenate((values, np.full(other.shape[1], '', dtype='object')))
                    else:
                        self.columnmeta[field] = np.concatenate((values, np.zeros(other.shape[1], dtype=values.dtype)))
            for field, values in other.columnmeta.items():
                if field not in self.columnmeta:
                    if values.dtype == 'object':
                        self.columnmeta[field] = np.concatenate((np.full(self.shape[1], '', dtype='object'), values))
                    else:
                        self.columnmeta[field] = np.concatenate((np.zeros(self.shape[1], dtype=values.dtype), values))
            self.updatesizeattribute()
            self.updateshapeattribute()
            self.updatedtypeattribute()
    
    def append(self, other, axis):
        if axis == 0:
            self.vertapp(other)
        elif axis == 1:
            self.horzapp(other)
        else:
            raise ValueError('invalid axis')
            
    def pop(self, tobepopped, axis):
        if type(tobepopped[0]) != np.bool_ and type(tobepopped[0]) != bool:
            raise TypeError('tobepopped must be numpy.bool_ or bool')
        elif tobepopped.sum() == 0:
            print('Nothing to be popped. Data matrix unchanged.')
        else:
            if axis==0:
                if tobepopped.size != self.shape[0]:
                    raise ValueError('size of tobepopped array does not match number of rows')
                else:
                    popped = datamatrix(rowname=self.rowname,
                                        rowlabels=np.full(tobepopped.sum(), '', dtype='object'),
                                        columnname=self.columnname,
                                        columnlabels=self.columnlabels.copy(),
                                        matrixname=self.matrixname,
                                        matrix=np.zeros((tobepopped.sum(),self.shape[1]), dtype='float64'),
                                        rowmeta={},
                                        columnmeta=copy.deepcopy(self.columnmeta))
                    popped.matrix = self.matrix[tobepopped,:]
                    self.matrix = self.matrix[~tobepopped,:]
                    popped.rowlabels = self.rowlabels[tobepopped]
                    self.rowlabels = self.rowlabels[~tobepopped]
                    for field, values in self.rowmeta.items():
                        popped.rowmeta[field] = values[tobepopped]
                        self.rowmeta[field] = values[~tobepopped]
            elif axis==1:
                if tobepopped.size != self.shape[1]:
                    raise ValueError('size of tobepopped array does not match number of columns')
                else:
                    popped = datamatrix(rowname=self.rowname,
                                        rowlabels=self.rowlabels.copy(),
                                        columnname=self.columnname,
                                        columnlabels=np.full(tobepopped.sum(), '', dtype='object'),
                                        matrixname=self.matrixname,
                                        matrix=np.zeros((self.shape[0],tobepopped.sum()), dtype='float64'),
                                        rowmeta=copy.deepcopy(self.rowmeta),
                                        columnmeta={})
                    popped.matrix = self.matrix[:,tobepopped]
                    self.matrix = self.matrix[:,~tobepopped]
                    popped.columnlabels = self.columnlabels[tobepopped]
                    self.columnlabels = self.columnlabels[~tobepopped]
                    for field, values in self.columnmeta.items():
                        popped.columnmeta[field] = values[tobepopped]
                        self.columnmeta[field] = values[~tobepopped]
            else:
                raise ValueError('invalid axis')
            self.updatesizeattribute()
            self.updateshapeattribute()
            self.updatedtypeattribute()
            popped.updatesizeattribute()
            popped.updateshapeattribute()
            popped.updatedtypeattribute()
            return popped
            
    def cluster_rows(self, metric='cosine', method='average'):
        if self.shape[0] == 0:
            raise ValueError('no rows in data matrix')
        elif self.shape[0] < 3:
            print('less than 3 rows. no clustering performed.')
        else:
            if self.shape[1] == 0:
                raise ValueError('no columns in data matrix')
            elif self.shape[1] == 1:
                self.reorder(np.argsort(self.matrix[:,0]), axis=0)
            else:
                tobepopped = np.all(self.matrix == 0, axis=1)
                if tobepopped.any() and metric in set(['cosine', 'correlation']):
                    popped = self.pop(tobepopped, axis=0)
                    self.reorder(hierarchy.leaves_list(fastcluster.linkage(self.matrix, method, metric)).astype('int64'), axis=0)
                    self.append(popped, axis=0)
                else:
                    self.reorder(hierarchy.leaves_list(fastcluster.linkage(self.matrix, method, metric)).astype('int64'), axis=0)
                
    def cluster_columns(self, metric='cosine', method='average'):
        if self.shape[1] == 0:
            raise ValueError('no columns in data matrix')
        elif self.shape[1] < 3:
            print('less than 3 columns. no clustering performed.')
        else:
            if self.shape[0] == 0:
                raise ValueError('no rows in data matrix')
            elif self.shape[0] == 1:
                self.reorder(np.argsort(self.matrix[0,:]), axis=1)
            else:
                tobepopped = np.all(self.matrix == 0, axis=0)
                if tobepopped.any() and metric in set(['cosine', 'correlation']):
                    popped = self.pop(tobepopped, axis=1)
                    self.reorder(hierarchy.leaves_list(fastcluster.linkage(self.matrix.T, method, metric)).astype('int64'), axis=1)
                    self.append(popped, axis=1)
                else:
                    self.reorder(hierarchy.leaves_list(fastcluster.linkage(self.matrix.T, method, metric)).astype('int64'), axis=1)
    
    def cluster_symmetric(self, method='average'):
        if self.shape[0] != self.shape[1]:
            raise ValueError('data matrix not square')
        elif self.shape[0] == 0:
            raise ValueError('no rows or columns in data matrix')
        elif self.shape[0] < 3:
            print('less than 3 rows and 3 columns. no clustering performed.')
        else:
            si = hierarchy.leaves_list(fastcluster.linkage(distance.squareform(np.float64(1) - self.matrix, checks=False), method)).astype('int64')
            self.reorder(si, axis=0)
            self.reorder(si, axis=1)

    def cluster(self, axis, metric='cosine', method='average'):
        if axis==0:
            self.cluster_rows(metric, method)
        elif axis==1:
            self.cluster_columns(metric, method)
        elif axis=='all':
            self.cluster_rows(metric, method)
            self.cluster_columns(metric, method)
        elif axis=='symmetric':
            self.cluster_symmetric(method)
        else:
            raise ValueError('invalid axis')
    
    def heatmap(self, rowmetalabels=None, columnmetalabels=None, normalize=False, standardize=False, normalizebeforestandardize=True, cmap_name='bwr', ub=None, lb=None, savefilename=None, closefigure=False, dpi=300):
        if type(cmap_name) == str:
            cmap_name = {'rowmeta':cmap_name, 'columnmeta':cmap_name, 'main':cmap_name}
        if type(rowmetalabels) == str and rowmetalabels != 'all':
            rowmetalabels = [rowmetalabels]
        if type(columnmetalabels) == str and columnmetalabels != 'all':
            columnmetalabels = [columnmetalabels]
        rowlabelcharlim = 10
        columnlabelcharlim = 40
        rowname = str(self.shape[0]) + ' ' + self.rowname
        columnname = str(self.shape[1]) + ' ' + self.columnname
        rowlabels = [x[:rowlabelcharlim]+'...' if len(x) > rowlabelcharlim else x for x in self.rowlabels]
        columnlabels = [x[:columnlabelcharlim]+'...' if len(x) > columnlabelcharlim else x for x in self.columnlabels]
        if rowmetalabels == 'all':
            rowmetalabels = list(self.rowmeta.keys())
        if columnmetalabels == 'all':
            columnmetalabels = list(self.columnmeta.keys())
        X = self.matrix.copy()
        
        fg, axs = plt.subplots(2, 2, figsize=(6.5,6.5))
        plt.delaxes(axs[0,0])
        if columnmetalabels == None or len(columnmetalabels) == 0:
            plt.delaxes(axs[0,1])
        else:
            columnmetamatrix = np.zeros((len(columnmetalabels), self.shape[1]), dtype='float64')
            for i, columnmetalabel in enumerate(columnmetalabels):
                columnmetamatrix[i,:] = self.columnmeta[columnmetalabel]
            rowmax = np.nanmax(np.abs(columnmetamatrix), axis=1)
            rowmax[rowmax==0] = 1
            columnmetamatrix = columnmetamatrix/rowmax[:,np.newaxis]
            columnmetalabels = [x[:rowlabelcharlim]+'...' if len(x) > rowlabelcharlim else x for x in columnmetalabels]
            ax = axs[0,1]
            ax.set_position([0.7/6.5, 6/6.5, 5/6.5, 0.4/6.5])
            ax.set_ylim(0, columnmetamatrix.shape[0])
            ax.set_xlim(0, columnmetamatrix.shape[1])
            ax.pcolormesh(columnmetamatrix, cmap=plt.get_cmap(cmap_name['columnmeta']), vmin=-1, vmax=1)
            ax.set_yticks(np.arange(columnmetamatrix.shape[0]) + 0.5, minor=False)
            ax.set_xticks(np.arange(columnmetamatrix.shape[1]) + 0.5, minor=False)
            ax.invert_yaxis()
            ax.set_yticklabels(columnmetalabels, minor=False, fontsize=8, fontname='arial')
            ax.grid(False)
            ax.tick_params(axis='x', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', pad=4)
            ax.tick_params(axis='y', which='major', left='off', right='off', labelleft='off', labelright='on', pad=4)
        if rowmetalabels == None or len(rowmetalabels) == 0:
            plt.delaxes(axs[1,0])
        else:
            rowmetamatrix = np.zeros((self.shape[0], len(rowmetalabels)), dtype='float64')
            for j, rowmetalabel in enumerate(rowmetalabels):
                rowmetamatrix[:,j] = self.rowmeta[rowmetalabel]
            colmax = np.nanmax(np.abs(rowmetamatrix), axis=0)
            colmax[colmax==0] = 1
            rowmetamatrix = rowmetamatrix/colmax[np.newaxis,:]
            rowmetalabels = [x[:columnlabelcharlim]+'...' if len(x) > columnlabelcharlim else x for x in rowmetalabels]
            ax = axs[1,0]
            ax.set_position([0.1/6.5, 2.8/6.5, 0.4/6.5, 3/6.5])
            ax.set_ylim(0, rowmetamatrix.shape[0])
            ax.set_xlim(0, rowmetamatrix.shape[1])
            ax.pcolormesh(rowmetamatrix, cmap=plt.get_cmap(cmap_name['rowmeta']), vmin=-1, vmax=1)
            ax.set_yticks(np.arange(rowmetamatrix.shape[0]) + 0.5, minor=False)
            ax.set_xticks(np.arange(rowmetamatrix.shape[1]) + 0.5, minor=False)
            ax.invert_yaxis()
            ax.set_xticklabels(rowmetalabels, minor=False, rotation=90, fontsize=8, fontname='arial')
            ax.grid(False)
            ax.tick_params(axis='x', which='major', bottom='off', top='off', labelbottom='on', labeltop='off', pad=4)
            ax.tick_params(axis='y', which='major', left='off', right='off', labelleft='off', labelright='off', pad=4)
        
        if normalizebeforestandardize:
            if normalize == 'columns':
                colmax = np.nanmax(np.abs(X), axis=0)
                colmax[colmax==0] = 1
                X = X/colmax[np.newaxis,:]
            elif normalize == 'rows':
                rowmax = np.nanmax(np.abs(X), axis=1)
                rowmax[rowmax==0] = 1
                X = X/rowmax[:,np.newaxis]
            if standardize == 'columns':
                colmean = np.nanmean(X, axis=0)
                colstdv = np.nanstd(X, axis=0)
                colstdv[colstdv==0] = 1
                X = (X - colmean[np.newaxis,:])/colstdv[np.newaxis,:]
            elif standardize == 'rows':
                rowmean = np.nanmean(X, axis=1)
                rowstdv = np.nanstd(X, axis=1)
                rowstdv[rowstdv==0] = 1
                X = (X - rowmean[:,np.newaxis])/rowstdv[:,np.newaxis]
        else:
            if standardize == 'columns':
                colmean = np.nanmean(X, axis=0)
                colstdv = np.nanstd(X, axis=0)
                colstdv[colstdv==0] = 1
                X = (X - colmean[np.newaxis,:])/colstdv[np.newaxis,:]
            elif standardize == 'rows':
                rowmean = np.nanmean(X, axis=1)
                rowstdv = np.nanstd(X, axis=1)
                rowstdv[rowstdv==0] = 1
                X = (X - rowmean[:,np.newaxis])/rowstdv[:,np.newaxis]
            if normalize == 'columns':
                colmax = np.nanmax(np.abs(X), axis=0)
                colmax[colmax==0] = 1
                X = X/colmax[np.newaxis,:]
            elif normalize == 'rows':
                rowmax = np.nanmax(np.abs(X), axis=1)
                rowmax[rowmax==0] = 1
                X = X/rowmax[:,np.newaxis]
        
        if ub == None:
            ub = np.nanmax(np.abs(X))
        if lb == None:
            lb = -ub
        
        ax = axs[1,1]
        ax.set_position([0.7/6.5, 2.8/6.5, 5/6.5, 3/6.5])
        ax.set_ylim(0, X.shape[0])
        ax.set_xlim(0, X.shape[1])
        ax.pcolormesh(X, cmap=plt.get_cmap(cmap_name['main']), vmin=lb, vmax=ub)
        ax.set_yticks(np.arange(X.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(X.shape[1]) + 0.5, minor=False)
        ax.invert_yaxis()
        ax.set_xticklabels(columnlabels, minor=False, rotation=90, fontsize=8*50/len(columnlabels), fontname='arial')
        ax.set_yticklabels(rowlabels, minor=False, fontsize=8*30/len(rowlabels), fontname='arial')
        ax.grid(False)
        ax.tick_params(axis='x', which='major', bottom='off', top='off', labelbottom='on', labeltop='off', pad=4)
        ax.tick_params(axis='y', which='major', left='off', right='off', labelleft='off', labelright='on', pad=4)
        ax.set_ylabel(rowname, fontsize=8, fontname='arial', labelpad=2)
        ax.set_title(columnname, loc='center', fontsize=8, fontname='arial')
        
        if savefilename != None and len(savefilename) > 0:
            plt.savefig(savefilename, transparent=True, pad_inches=0, dpi=dpi)
        
        if closefigure:
            plt.close()
        else:
            plt.show()

    
    def merge(self, axis, merge_function=np.nanmean):
        if axis==0:
            labels = self.rowlabels
            meta = self.rowmeta
            matrix = self.matrix
        elif axis==1:
            labels = self.columnlabels
            meta = self.columnmeta
            matrix = self.matrix.T
        else:
            raise ValueError('invalid axis')
        ulabels = np.unique(labels)
        n = ulabels.size
        if n < labels.size:
            umeta = {}
            for metaname in meta:
                umeta[metaname] = np.zeros(n, dtype=meta[metaname].dtype)
            umatrix = np.zeros((n, matrix.shape[1]), dtype='float64')
            for i, label in enumerate(ulabels):
                hit = labels == label
                for metaname, metadata in meta.items():
                    if metadata.dtype == 'object':
                        umeta[metaname][i] = '|'.join(np.unique(metadata[hit]))
                    else:
                        umeta[metaname][i] = merge_function(metadata[hit])
                umatrix[i,:] = merge_function(matrix[hit,:], axis=0)
            if axis==0:
                self.rowlabels = ulabels
                self.rowmeta = umeta
                self.matrix = umatrix
            else:
                self.columnlabels = ulabels
                self.columnmeta = umeta
                self.matrix = umatrix.T
            self.updatesizeattribute()
            self.updateshapeattribute()
            self.updatedtypeattribute()
        else:
            print('Data matrix already unique along axis={!s}. Data matrix unchanged.'.format(axis))
    
    def nanreplace(self, axis, replace_function=np.nanmean):
        is_nan = np.isnan(self.matrix)
        if axis==-1:
            if replace_function == 'bootstrap':
                print('WARNING: bootstrap does not make sense if correlation structure in data matrix is relevant for subsequent analysis.')
                self.matrix[is_nan] = np.random.choice(self.matrix[~is_nan], is_nan.sum(), replace=True)
            else:
                self.matrix[is_nan] = replace_function(self.matrix)
        else:
            if replace_function == 'bootstrap':
                print('WARNING: bootstrap does not make sense if correlation structure in data matrix is relevant for subsequent analysis.')
                if axis == 0:
                    for i in range(self.shape[1]):
                        self.matrix[is_nan[:,i],i] = np.random.choice(self.matrix[~is_nan[:,i],i], is_nan[:,i].sum(), replace=True)
                else:
                    for i in range(self.shape[0]):
                        self.matrix[i,is_nan[i,:]] = np.random.choice(self.matrix[i,~is_nan[i,:]], is_nan[i,:].sum(), replace=True)
            else:
                replace_values = replace_function(self.matrix, axis=axis, keepdims=True)
                self.matrix[is_nan] = 0
                self.matrix += replace_values*is_nan
            
    def select(self, rowlabels, columnlabels):
        if rowlabels == []:
            submatrix = self.matrix
        else:
            if type(rowlabels) != list and type(rowlabels) != np.ndarray:
                rowlabels = [rowlabels]
            if ~np.all(np.in1d(rowlabels, self.rowlabels)):
                raise ValueError('not all rowlabels in datamatrix')
            rowindices = np.array([(self.rowlabels==label).nonzero()[0][0] for label in rowlabels], dtype='int64')
            submatrix = self.matrix[rowindices,:]
        if columnlabels == []:
            if rowlabels == []:
                raise ValueError('nothing selected.')
            else:
                pass
        else:
            if type(columnlabels) != list and type(columnlabels) != np.ndarray:
                columnlabels = [columnlabels]
            if ~np.all(np.in1d(columnlabels, self.columnlabels)):
                raise ValueError('not all columnlabels in datamatrix')
            columnindices = np.array([(self.columnlabels==label).nonzero()[0][0] for label in columnlabels], dtype='int64')
            submatrix = submatrix[:,columnindices]
        if submatrix.size == 1:
            return submatrix[0,0]
        elif submatrix.shape[0] == 1 or submatrix.shape[1] == 1:
            return submatrix.reshape(-1)
        else:
            return submatrix
        
    def totranspose(self):
        return datamatrix(rowname=self.columnname,
                          rowlabels=self.columnlabels.copy(),
                          columnname=self.rowname,
                          columnlabels=self.rowlabels.copy(),
                          matrixname=self.matrixname,
                          matrix=self.matrix.T.copy(),
                          rowmeta=copy.deepcopy(self.columnmeta),
                          columnmeta=copy.deepcopy(self.rowmeta))

    def tolabels(self, rowlabels=[], columnlabels=[], fillvalue='zero'):
        if type(rowlabels) == str:
            rowlabels = [rowlabels]
        if type(rowlabels) == list:
            rowlabels = np.array(rowlabels, dtype='object')
        if type(columnlabels) == str:
            columnlabels = [columnlabels]
        if type(columnlabels) == list:
            columnlabels = np.array(columnlabels, dtype='object')
        if rowlabels.size == 0 or np.all(rowlabels == self.rowlabels):
            rowlabels = self.rowlabels.copy()
            newrowmeta = copy.deepcopy(self.rowmeta)
            submatrix = self.matrix
        else:
            newhitindices = np.in1d(rowlabels, self.rowlabels).nonzero()[0]
            newhitlabels = rowlabels[newhitindices]
            si = np.argsort(newhitlabels)
            newhitlabels = newhitlabels[si]
            newhitindices = newhitindices[si]
            oldhitindices = np.in1d(self.rowlabels, rowlabels).nonzero()[0]
            oldhitlabels = self.rowlabels[oldhitindices]
            si = np.argsort(oldhitlabels)
            oldhitlabels = oldhitlabels[si]
            oldhitindices = oldhitindices[si]
            submatrix = np.zeros((rowlabels.size, self.shape[1]), dtype=self.dtype) if fillvalue == 'zero' \
                   else np.full((rowlabels.size, self.shape[1]), fillvalue, dtype=self.dtype)
            submatrix[newhitindices,:] = self.matrix[oldhitindices,:]
            newrowmeta = {}
            for field, values in self.rowmeta.items():
                if values.dtype == np.object:
                    newrowmeta[field] = np.full(rowlabels.size, '', dtype=values.dtype)
                else:
                    newrowmeta[field] = np.zeros(rowlabels.size, dtype=values.dtype)
                newrowmeta[field][newhitindices] = values[oldhitindices]
        if columnlabels.size == 0 or np.all(columnlabels == self.columnlabels):
            columnlabels = self.columnlabels.copy()
            newcolumnmeta = copy.deepcopy(self.columnmeta)
            newmatrix = submatrix
        else:
            newhitindices = np.in1d(columnlabels, self.columnlabels).nonzero()[0]
            newhitlabels = columnlabels[newhitindices]
            si = np.argsort(newhitlabels)
            newhitlabels = newhitlabels[si]
            newhitindices = newhitindices[si]
            oldhitindices = np.in1d(self.columnlabels, columnlabels).nonzero()[0]
            oldhitlabels = self.columnlabels[oldhitindices]
            si = np.argsort(oldhitlabels)
            oldhitlabels = oldhitlabels[si]
            oldhitindices = oldhitindices[si]
            newmatrix = np.zeros((rowlabels.size, columnlabels.size), dtype=self.dtype) if fillvalue == 'zero' \
                   else np.full((rowlabels.size, columnlabels.size), fillvalue, dtype=self.dtype)
            newmatrix[:,newhitindices] = submatrix[:,oldhitindices]
            newcolumnmeta = {}
            for field, values in self.columnmeta.items():
                if values.dtype == np.object:
                    newcolumnmeta[field] = np.full(columnlabels.size, '', dtype=values.dtype)
                else:
                    newcolumnmeta[field] = np.zeros(columnlabels.size, dtype=values.dtype)
                newcolumnmeta[field][newhitindices] = values[oldhitindices]
        return datamatrix(rowname=self.rowname,
                          rowlabels=rowlabels,
                          columnname=self.columnname,
                          columnlabels=columnlabels,
                          matrixname=self.matrixname,
                          matrix=newmatrix,
                          rowmeta=newrowmeta,
                          columnmeta=newcolumnmeta)
    
    def tosimilarity(self, axis, metric='cosine'):
        if axis==0:
            return datamatrix(rowname=self.rowname,
                              rowlabels=self.rowlabels.copy(),
                              columnname=self.rowname,
                              columnlabels=self.rowlabels.copy(),
                              matrixname=self.rowname + '-' + self.rowname + '_' + metric + '_similarity_derived_from_' + self.matrixname,
                              matrix=mlstats.corr(self.matrix, axis=axis, metric=metric, getpvalues=False),
                              rowmeta=copy.deepcopy(self.rowmeta),
                              columnmeta=copy.deepcopy(self.rowmeta))
        elif axis==1:
            return datamatrix(rowname=self.columnname,
                              rowlabels=self.columnlabels.copy(),
                              columnname=self.columnname,
                              columnlabels=self.columnlabels.copy(),
                              matrixname=self.columnname + '-' + self.columnname + '_' + metric + '_similarity_derived_from_' + self.matrixname,
                              matrix=mlstats.corr(self.matrix, axis=axis, metric=metric, getpvalues=False),
                              rowmeta=copy.deepcopy(self.columnmeta),
                              columnmeta=copy.deepcopy(self.columnmeta))
        else:
            raise ValueError('invalid axis')
    
    def tosimilaritypvalues(self, axis, metric='cosine'):
        if axis==0:
            return datamatrix(rowname=self.rowname,
                              rowlabels=self.rowlabels.copy(),
                              columnname=self.rowname,
                              columnlabels=self.rowlabels.copy(),
                              matrixname=self.rowname + '-' + self.rowname + '_' + metric + '_similarity_derived_from_' + self.matrixname,
                              matrix=mlstats.corr(self.matrix, axis=axis, metric=metric, getpvalues=True)[1],
                              rowmeta=copy.deepcopy(self.rowmeta),
                              columnmeta=copy.deepcopy(self.rowmeta))
        elif axis==1:
            return datamatrix(rowname=self.columnname,
                              rowlabels=self.columnlabels.copy(),
                              columnname=self.columnname,
                              columnlabels=self.columnlabels.copy(),
                              matrixname=self.columnname + '-' + self.columnname + '_' + metric + '_similarity_derived_from_' + self.matrixname,
                              matrix=mlstats.corr(self.matrix, axis=axis, metric=metric, getpvalues=True)[1],
                              rowmeta=copy.deepcopy(self.columnmeta),
                              columnmeta=copy.deepcopy(self.columnmeta))
        else:
            raise ValueError('invalid axis')
