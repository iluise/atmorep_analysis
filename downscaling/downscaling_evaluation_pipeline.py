# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC), European Centre for Medium-Range Weather Forecasts (ECMWF), 
#                              European Organization for Nuclear Research (CERN) - IT
#
# SPDX-License-Identifier: MIT

__authors__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2024-12-16"
__update__ = "2025-01-13"

"""
Main script to run evaluation pipeline for downscaling application.
"""

# import packages
from pathlib import Path
import glob
from typing import Union, List, Dict
from functools import partial
import numpy as np
import xarray as xr

from downscaling_evaluation_utils import eval_deterministic_forecast, eval_probablistic_forecast, _extract_ens_mem, get_month_from_nc_fname

str_or_path = Union[str, Path]

# main function
def main(parser_args):

    # get parsed arguments
    datadir = parser_args.datadir #Path("/p/scratch/hclimrep/pavel1/Public/HarrisWGAN/n0_2-2_16575_e16")
    outdir = parser_args.outdir #Path("/p/scratch/hclimrep/michael1/evaluation")
    data_format = parser_args.data_format
    ens = parser_args.ens
    quick_evaluate = parser_args.quick_evaluate
    nsamples_quick = parser_args.nsamples_quick
    seed = parser_args.seed

    # set seed
    np.random.seed(seed)

    # read data depending on data format
    if data_format == "netcdf":
        flist = glob.glob(str(datadir.joinpath("pred_samples*.nc")))
        flist = sorted(flist, key=get_month_from_nc_fname)

        if ens == "mean":
            opt_preprocess = {"drop_variables": "tot_prec_pred"}   # don't require ensemble information
            var_fcst = "tot_prec_pred_mean"
        elif ens == "all":
            opt_preprocess = {"drop_variables": "tot_prec_pred_mean"}
            var_fcst = "tot_prec_pred"
        elif isinstance(ens, int):
            opt_preprocess = {"drop_variables": "tot_prec_pred_mean", "preprocess": partial(_extract_ens_mem, ens_mem=ens)}
            var_fcst = "tot_prec_pred"
        else:
            raise ValueError(f"Invalid ensemble value chosen. Choose one of the following: 'mean', 'all', <int>")
            

        ds = xr.open_mfdataset(flist, **opt_preprocess, chunks={"time": 4})
        ntimes = len(ds["time"])    
    elif data_format == "zarr":
        raise NotImplementedError("Zarr format not yet implemented.")
    

    # reduce number of samples for quick evaluation
    if quick_evaluate:
        tidx = sorted(np.random.choice(np.arange(ntimes), nsamples_quick, replace=False))
        ds = ds.isel({"time": tidx})#.load()
        tidx = np.arange(len(tidx))

    # run evaluation depending on ensemble type
    if ens == "mean" or isinstance(ens, int):
        eval_deterministic_forecast(ds[var_fcst], ds["tot_prec_ref"], outdir=outdir)
    else:
        eval_probablistic_forecast(ds[var_fcst], ds["tot_prec_ref"], outdir=outdir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation pipeline for downscaling application.")
    parser.add_argument("--datadir", type=Path, help="Directory containing forecast data.")
    parser.add_argument("--outdir", type=Path, help="Directory to store evaluation results.")
    parser.add_argument("--data_format", type=str, default="netcdf", help="Format of forecast data. Choose between 'netcdf' and 'zarr'.")
    parser.add_argument("--ens", type=Union[str, int], default="all", help="Type of ensemble to evaluate. Choose between 'mean', 'all', <int>.")
    parser.add_argument("--quick_evaluate", action="store_true", help="Quick evaluation with reduced number of samples.")
    parser.add_argument("--nsamples_quick", type=int, default=256, help="Number of samples for quick evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator.")

    args = parser.parse_args()
    main(args)