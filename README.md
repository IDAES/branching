Branching
==============================

Selin Bayramoglu, Georgia Institute of Technology

Nick Sahinidis, Georgia Institute of Technology

Python package to build sparse regression models of strong branching and use them for branching. These models are the lasso, and L0 and L1/L2 penalized linear regression.

### Copyright

Copyright (c) 2023, Selin Bayramoglu


#### Acknowledgements
 
Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

## Languages

This project uses Python3, R and C.
## Conda environment

We suggest that users create a new conda environment to install this package.

`conda create -n branching-env python=3.9`

`conda activate branching-env`


## SCIP

This project uses the open-source solver SCIP 8.0.0. To install it, run

`conda install -c conda-forge scip=8.0.0`

## Ecole

For collecting training data via Python, we use a modified version of the [Ecole library](https://github.com/ds4dm/ecole) 0.8.1. To install it with the currently installed SCIP version, run

`cd ecole`

`CMAKE_ARGS="-DSCIP_DIR=/Users/selinbayramoglu/opt/anaconda3/envs/branching-env/lib/cmake/scip -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_BUILD_TYPE=Release" python -m pip install .`

Change SCIP_DIR in the above command to the following path:

`/Path/to/branching-env/lib/cmake/scip`

## R packages

We use the R packages [`glmnet`](https://glmnet.stanford.edu/) and [`l0learn`](https://cran.r-project.org/web/packages/L0Learn/vignettes/L0Learn-vignette.html) for building models. They can be installed as follows:

`install.packages("glmnet", repos = "https://cran.us.r-project.org")`

`install.packages("L0Learn", repos = "http://cran.rstudio.com")`


## Branching

To install this package and its dependencies locally, do

`pip install -r requirements.txt`

## Instances

Instances for training, validation and evaluation are in the `instances-setcover` folder. These are set covering problems generated randomly as in https://github.com/ds4dm/learn2branch-ecole .

### Testing a sparse model

Once a sparse branching rule, e.g. glmnet_lasso, is learned, and a model file is created, you can test the model's performance as follows:

In line 5 of `branching/test-methods/CMakeLists.txt`, set SCIP_DIR to `/Path/to/branching-env/lib/cmake/scip` 

`cd branching/test-methods/build`

`cmake .. -DCMAKE_BUILD_TYPE=Release`

`make`

(Note: If you experience a linking problem during compilation, do:
`export DYLD_LIBRARY_PATH=/Path/to/branching-env/lib`)

This builds an executable called `sparse`. To run this rule for branching on an instance, e.g. instance_1.lp, with seed 0:

`./sparse ../../../instances-setcover/evaluation/instance_1.lp glmnet_lasso 0`

The branching rules available in SCIP, such as the default rule `relpscost`, can be tested similarly:

`./sparse ../../../instances-setcover/evaluation/instance_1.lp relpscost 0`


### Running the example

First, run `example-part1-data.ipynb` to collect data and create datasets. Once this runs correctly, do

`bash example-part2-model-building.sh`

to build sparse models for braching.

Finally, do

`bash example-part3-testing.sh`

to run the learned models and the default SCIP on the evaluation problems.





