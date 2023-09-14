#!/bin/bash

cd branching/test-methods/build
export DYLD_LIBRARY_PATH=/Users/selinbayramoglu/opt/anaconda3/envs/branching-env/lib
## Test the sparse models and SCIP's default rule relpscost on the evaluation instances:

for id in {1..20}
do
./sparse ../../../instances-setcover/evaluation/instance_${id}.lp glmnet_lasso 0
./sparse ../../../instances-setcover/evaluation/instance_${id}.lp l0learn_l0l1 0
./sparse ../../../instances-setcover/evaluation/instance_${id}.lp l0learn_l0l2 0

./sparse ../../../instances-setcover/evaluation/instance_${id}.lp relpscost 0
done