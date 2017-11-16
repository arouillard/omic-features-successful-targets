# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import os
import numpy as np
import datasetIO

validation_reps = 200
validation_folds = 5

# iterate over cross-validation reps and folds
print('iterating over cross-validation reps and folds...', flush=True)
with open('datasets/useful_features/inner_crossvalidation_summary.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
    writelist = ['validation_rep',
                 'validation_fold',
                 'rf_model_name',
                 'rf_num_features',
                 'rf_auroc_mean',
                 'rf_auroc_stdv',
                 'rf_auprc_mean',
                 'rf_auprc_stdv',
                 'lr_model_name',
                 'lr_num_features',
                 'lr_auroc_mean',
                 'lr_auroc_stdv',
                 'lr_auprc_mean',
                 'lr_auprc_stdv']
    fw.write('\t'.join(writelist) + '\n')
    
    for validation_rep in range(validation_reps):
        for validation_fold in range(validation_folds):
            stat_model_path = 'datasets/useful_features/rep{0!s}_fold{1!s}/stat_model.txt.gz'.format(validation_rep, validation_fold)        
            if os.path.exists(stat_model_path):
                print('working on rep {0!s} fold {1!s}'.format(validation_rep, validation_fold), flush=True)
                
                # load inner loop performance statsfor all models
                print('    loading inner loop performance stats for all models...', flush=True)
                stat_model = datasetIO.load_datamatrix(stat_model_path)
                
                # select simplest model (fewest features) with auroc and auprc within 95% of max
                print('    selecting simplest model (fewest features) with auroc and auprc within 95% of max...', flush=True)
                model_scores = 0.5*(stat_model.select('auroc_mean',[]) + stat_model.select('auprc_mean',[]))
                rfmodel_hit = np.logical_and(model_scores >= 0.95*model_scores.max(), stat_model.columnmeta['model_type']=='random_forest')
                if ~rfmodel_hit.any():
                    rfmodel_hit = np.logical_and(model_scores >= 0.99*model_scores[stat_model.columnmeta['model_type']=='random_forest'].max(), stat_model.columnmeta['model_type']=='random_forest')
                selected_rfmodel_index = np.where(rfmodel_hit)[0][-1]
                selected_rfmodel_name = stat_model.columnlabels[selected_rfmodel_index]
                lrmodel_hit = np.logical_and(model_scores >= 0.95*model_scores.max(), stat_model.columnmeta['model_type']=='logistic_regression')
                if ~lrmodel_hit.any():
                    lrmodel_hit = np.logical_and(model_scores >= 0.99*model_scores[stat_model.columnmeta['model_type']=='logistic_regression'].max(), stat_model.columnmeta['model_type']=='logistic_regression')
                selected_lrmodel_index = np.where(lrmodel_hit)[0][-1]
                selected_lrmodel_name = stat_model.columnlabels[selected_lrmodel_index]
                
                # write crossvalidation performance stats for selected rf and lr models
                print('    writing crossvalidation performance stats for selected rf and lr models...', flush=True)
                writelist = ['{0:1.3g}'.format(validation_rep),
                             '{0:1.3g}'.format(validation_fold),
                             selected_rfmodel_name,
                             stat_model.columnmeta['num_features'][selected_rfmodel_index],
                             '{0:1.3g}'.format(stat_model.select('auroc_mean',selected_rfmodel_name)),
                             '{0:1.3g}'.format(stat_model.select('auroc_stdv',selected_rfmodel_name)),
                             '{0:1.3g}'.format(stat_model.select('auprc_mean',selected_rfmodel_name)),
                             '{0:1.3g}'.format(stat_model.select('auprc_stdv',selected_rfmodel_name)),
                             selected_lrmodel_name,
                             stat_model.columnmeta['num_features'][selected_lrmodel_index],
                             '{0:1.3g}'.format(stat_model.select('auroc_mean',selected_lrmodel_name)),
                             '{0:1.3g}'.format(stat_model.select('auroc_stdv',selected_lrmodel_name)),
                             '{0:1.3g}'.format(stat_model.select('auprc_mean',selected_lrmodel_name)),
                             '{0:1.3g}'.format(stat_model.select('auprc_stdv',selected_lrmodel_name))]
                fw.write('\t'.join(writelist) + '\n')

print('done.', flush=True)
