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

import get_generalizable_features
import get_merged_features
import get_useful_features

def main(validation_rep=0, validation_fold=0):
    
    print('VALIDATION_REP: {0!s}, VALIDATION_FOLD:{1!s}'.format(validation_rep, validation_fold), flush=True)

    print('GETTING GENERALIZABLE FEATURES...', flush=True)
    get_generalizable_features.main(validation_rep, validation_fold)
    
    print('GETTING MERGED FEATURES...', flush=True)
    get_merged_features.main(validation_rep, validation_fold)
    
    print('GETTING USEFUL FEATURES...', flush=True)
    get_useful_features.main(validation_rep, validation_fold)

if __name__ == '__main__':
    main(validation_rep=int(sys.argv[1]), validation_fold=int(sys.argv[2]))
