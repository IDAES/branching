import os
import argparse
import numpy as np
import pandas as pd
import pathlib

import natsort

import branching
from branching.utilities import log, load_flat_samples

def load_samples(filenames, size_limit, logfile=None):

    """
    Function to read and store the candidate measurements into a single dataset.


    Parameters
    ----------
    filenames : list of str
        List of the names of the sample files.

    size_limit : int
        Maximum number of candidate measurements.

    Returns
    -------
    x : array-like
        2-D array of the measurements.
    y : array-like
        1-D array of the scores.
    ncands : int
        Number of candidate measurements in the dataset.
    """


    x, y, ncands = [], [], []
    total_ncands = 0

    for i, filename in enumerate(filenames):
        cand_x, cand_y, _ = load_flat_samples(filename)
        sum_x = sum(sum(np.isnan(cand_x)))
        sum_y = sum(np.isnan(cand_y))
        cand_x = np.nan_to_num(cand_x)
        
        x.append(cand_x)
        y.append(cand_y)
        ncands.append(cand_x.shape[0])
        total_ncands += ncands[-1]

        if (i + 1) % 100 == 0:
            log(f"  {i+1}/{len(filenames)} files processed ({total_ncands} candidate variables)", logfile)

        if total_ncands >= size_limit:
            log(f"  dataset size limit reached ({size_limit} candidate variables)", logfile)
            break

    x = np.concatenate(x)
    y = np.concatenate(y)
    ncands = np.asarray(ncands)

    if total_ncands > size_limit:
        x = x[:size_limit]
        y = y[:size_limit]
        ncands[-1] -= total_ncands - size_limit

    return x, y, ncands
