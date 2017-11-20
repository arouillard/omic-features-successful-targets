# Systematic interrogation of diverse Omic data reveals interpretable, robust, and generalizable transcriptomic features of clinically successful therapeutic targets
### Andrew D. Rouillard, Mark R. Hurle, and Pankaj Agarwal
### https://www.biorxiv.org/content/early/2017/11/16/220848

## Contents

### Scripts

#### get_candidate_features.py
For each Harmonizome dataset--formatted as a matrix with genes labeling the rows and features labeling the columns--append mean and standard deviation as additional features/columns, map phase III clinical trial outcomes to genes, discard features lacking information about genes with known phase III clinical trial outcomes, and save processed dataset in folder "datasets/candidate_features".

#### get_nonredundant_features.py
For each dataset in "datasets/candidate_features", perform dimensionality reduction and save processed dataset in folder "datasets/nonredundant_features".

#### get_significant_features.py
For each dataset in "datasets/nonredundant_features", perform permutation tests to obtain p-values for each feature indicating the significance of the difference between targets that have succeeded versus failed in phase III clinical trials, apply multiple hypothesis testing correction, write nominal and corrected p-values to file, and save dataset with significant features in folder "datasets/significant_features".

#### get_generalizable_features.py
For each dataset in "datasets/nonredundant_features", perform permutation tests--only shuffling success/failure labels within target classes to control for target class as a confounding factor--to obtain p-values for each feature indicating the significance of the difference between targets that have succeeded or failed in phase III clinical trials, apply multiple hypothesis testing correction, write nominal and corrected p-values to file, and save dataset with significant features in folder "datasets/generalizable_features/repX_foldY". Significance testing only considers training examples in a specified cross-validation repetition X and fold Y.

#### get_merged_features.py
Merge features from datasets in "datasets/generalizable_features/repX_foldY" into a single feature matrix and save merged dataset in folder "datasets/merged_features/repX_foldY".

#### get_useful_features.py
Load feature matrix in "datasets/merged_features/repX_foldY", perform incremental feature elimination using an inner cross-validation loop to select a classifier type (Random Forest or logistic regression) and subset of features, choosing the simplest model with inner loop cross-validation AUROC and AUPR within 95% of max. Refit selected model to all training examples for outer loop cross-validation repetition X and fold Y and obtain predictions for test examples and unlabeled examples. Save inner loop cross-validation data, selected model parameters and performance statistics, and model predictions in "datasets/useful_features/repX_foldY".

#### crossvalidation_pipeline.py
Runs get_generalizable_features.py, get_merged_features.py, and get_useful_features.py for a specified cross-validation repetition X and fold Y.

#### collect_significant_results.py
Collect feature significance testing results from all datasets into a single file.

#### collect_generalizable_results.py
Collect significant features (from permutation test controlling for target class as a confounding factor) from all datasets and all 1000 cross-validation instances (200 repetitions x 5 folds) into a single file.

#### collect_inner_crossvalidation_results.py
Collect inner loop cross-validation AUROC and AUPR values for Random Forest versus logistic regression for all 1000 cross-validation instances.

#### collect_crossvalidation_results.py
Collect cross-validation classifier properties (model type and set of features), classifier performance statistics, and classifier predictions for all 1000 cross-validation instances.

#### get_validation_examples.py
For each of 200 repetitions, randomly assign targets with phase III clinical trial outcomes to 5 folds, then for each of 5 folds, write list of test examples to "targets/validation_examples/repX_foldY.txt"

#### get_target_clusters.py
Assign targets with phase III clinical trial outcomes to clusters based on their membership in HGNC gene families. Save cluster assignments to "targets/clusters/gene_cluster_byfamily.pickle"

### Folders

#### datasets
Input and/or output datasets for each stage of analysis, as described in Scripts.

#### results
Results figures and tables for the paper.

#### targets
Phase III clinical trial outcomes, assignments to target classes, and assignments to test sets.
