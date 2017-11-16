# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import os

# collect feature significance for all datasets
print('collecting feature significance for all datasets...', flush=True)
with open('datasets/significant_features/significant_summary.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
    filenames = [x for x in os.listdir('datasets/significant_features') if '_feature_significance_info.txt' in x]
    for i, filename in enumerate(filenames):
        print('    working on file:{0}'.format(filename), flush=True)
        with open('datasets/significant_features/{0}'.format(filename), mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
            headerline = fr.readline()                    
            if i == 0:
                fw.write(headerline)
            for line in fr:
                fw.write(line)
print('done.', flush=True)
