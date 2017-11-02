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
from sklearn.model_selection import StratifiedKFold

def main():
    
    # load class examples
    print('loading class examples...', flush=True)
    class_examples_folder = 'targets/pharmaprojects'
    class_examples = {'positive':datasetIO.load_examples('{0}/positive.txt'.format(class_examples_folder)),
                      'negative':datasetIO.load_examples('{0}/negative.txt'.format(class_examples_folder)),
                      'unknown':datasetIO.load_examples('{0}/unknown.txt'.format(class_examples_folder))}
    
    # create gene and class label arrays
    print('creating gene and class label arrays...', flush=True)
    G = np.array(list(class_examples['positive'])+list(class_examples['negative']), dtype='object')
    Y = np.append(np.ones(len(class_examples['positive']), dtype='bool'), np.zeros(len(class_examples['negative']), dtype='bool'))
    
    # specify cross-validation parameters
    print('specifying cross-validation parameters...', flush=True)
    reps = 200
    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    
    # specify results folder
    print('specifying results folder...', flush=True)
    results_folder = 'targets/validation_examples'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    
    # generate validation examples
    print('generating validation examples...', flush=True)
    for rep in range(reps):
        for fold, (traintest_indices, valid_indices) in enumerate(skf.split(G, Y)):
            validation_examples = G[valid_indices]
            print('saving validation examples for fold {0!s} of rep {1!s}...'.format(fold, rep), flush=True)
            with open('{0}/rep{1!s}_fold{2!s}.txt'.format(results_folder, rep, fold), mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
                fw.write('\n'.join(validation_examples))
    
    print('done.', flush=True)

if __name__ == '__main__':
    main()
