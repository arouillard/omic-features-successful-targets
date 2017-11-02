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

validation_reps = 200
validation_folds = 5

# collect significant features for all cross-validation reps and folds
print('collecting significant features for all cross-validation reps and folds...', flush=True)
with open('datasets/generalizable_features/generalizable_summary.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
    for validation_rep in range(validation_reps):
        for validation_fold in range(validation_folds):
            print('    working on validation_rep:{0!s}, validation_fold:{1!s}'.format(validation_rep,validation_fold), flush=True)
            filenames = [x for x in os.listdir('datasets/generalizable_features/rep{0!s}_fold{1!s}'.format(validation_rep, validation_fold)) if '_feature_generalizability_info.txt' in x]
            for i, filename in enumerate(filenames):
                print('        working on file:{0}'.format(filename), flush=True)
                with open('datasets/generalizable_features/rep{0!s}_fold{1!s}/{2}'.format(validation_rep, validation_fold, filename), mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
                    headerline = fr.readline()                    
                    if validation_rep == 0 and validation_fold == 0 and i == 0:
                        fw.write('validation_rep\tvalidation_fold\t' + headerline)
                    for line in fr:
                        entries = [x.strip() for x in line.split('\t')]
                        if entries[6] == '1': # if significant
                            fw.write('{0!s}\t{1!s}\t'.format(validation_rep, validation_fold) + line)
print('done.', flush=True)
