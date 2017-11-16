# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import sys
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
