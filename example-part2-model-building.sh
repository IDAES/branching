#!/bin/bash

cd branching

## To build a single model for dataset 0 using the sparse regression methods, we do:

Rscript build_sparse_models.r 0 glmnet_lasso
Rscript build_sparse_models.r 0 l0learn_l0l1
Rscript build_sparse_models.r 0 l0learn_l0l2

