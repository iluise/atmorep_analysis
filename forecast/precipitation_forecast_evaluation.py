# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC), European Centre for Medium-Range Weather Forecasts (ECMWF), 
#                              European Organization for Nuclear Research (CERN) - IT
#
# SPDX-License-Identifier: MIT

__authors__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2025-01-20"
__update__ = "2025-01-20"

"""
Main script to run evaluation pipeline for precipitation forecasts.
"""

# import packages
from pathlib import Path
import glob
from typing import Union, List, Dict
from functools import partial
import numpy as np
import xarray as xr

from scripts.precipitation_forecast_evaluation_utils import eval_deterministic_forecast, eval_probablistic_forecast
from utils.read_atmorep_data import HandleAtmoRepData

str_or_path = Union[str, Path]

# main function
def main(parser_args):

    # get parsed arguments
    datadir = parser_args.datadir #Path("/p/scratch/hclimrep/pavel1/Public/HarrisWGAN/n0_2-2_16575_e16")
    model_id = parser_args.model_id
    outdir = parser_args.outdir #Path("/p/scratch/hclimrep/michael1/evaluation")
    ens = parser_args.ens
    quick_evaluate = parser_args.quick_evaluate
    nforecasts_quick = parser_args.nforecasts_quick
    seed = parser_args.seed

    # set seed
    np.random.seed(seed)

    # get data reader...
    ar_data = HandleAtmoRepData(model_id, str(datadir))

    # ... and read data
    tot_prec_tar = ar_data.read_data("total_precip", "target")
    tot_prec_tar.name = "total_precipitation_era5"

    if ens == "mean":
        tot_prec_pred = ar_data.read_data("total_precip", "pred")
        tot_prec_pred.name = "total_precipitation_atmorep_mean"
    else:   
        tot_prec_pred = ar_data.read_data("total_precip", "ens")
        tot_prec_pred.name = "total_precipitation_atmorep"

    nforecasts = tot_prec_pred["init_time"].size
    # remove redundant ml-dimension and convert to mm/h
    tot_prec_tar = tot_prec_tar.squeeze()*1000.    # .isel({"lon": slice(216)})
    tot_prec_pred = tot_prec_pred.squeeze()*1000.    # .isel({"lon": slice(216)})

    if isinstance(ens, int):
        tot_prec_pred = tot_prec_pred.isel({"ensemble": ens})

    # To-Do: only for forecastst over limited area -> avoid hard-coding in the future
    lon = tot_prec_pred["lon"]
    tot_prec_pred["lon"] = lon.where(lon < 180., lon-360.)
    tot_prec_tar["lon"] = lon.where(lon < 180., lon-360.)  

    # read data depending on data format
    if ens == "all" or isinstance(ens, int):
        tot_prec_pred = tot_prec_pred.transpose("init_time", "lead_time", "lat", "lon", "ensemble")

    # reduce number of samples for quick evaluation
    nforecasts_quick = min(nforecasts, nforecasts_quick)
    if quick_evaluate:
        tidx = sorted(np.random.choice(np.arange(nforecasts), nforecasts_quick, replace=False))
        ds = ds.isel({"init_time": tidx})
        tidx = np.arange(len(tidx))

    # run evaluation depending on ensemble type
    if ens == "mean" or isinstance(ens, int):
        eval_func = eval_deterministic_forecast
        # To-Do: read from config-file
        eval_dict = {"scores": {"thresh_ets": [.1, .5, 1.]}, 
                     "histogram": {"legend_labels": ["AtmoRep mean", "ERA 5"],
                                   "bins": np.array([-1, 0., .01, 0.1, 0.5, 1., 2., 5., 10., 15., 20]), "figsize": (12, 8)}, 
                     "comparison_map": {"nsamples_plot": 10, "nens": 3, "cmap_name": "PuOr", "cmap_range": (.5, 1.), 
                                        "levels": [0., 0.25, 0.5, 1., 1.5, 2.5, 5., 7.5, 10., 15., 20., 30., 50., 75.]}
                    }       
    else:
        eval_func = eval_probablistic_forecast
        # To-Do: read from config-file
        eval_dict = {"scores": {"noise_fac": 1.e-03,}, 
                     "histogram": {"legend_labels": ["AtmoRep", "ERA 5"],
                                   "bins": np.array([-1, 0., .01, 0.1, 0.5, 1., 2., 5., 10., 15., 20]), "figsize": (12, 8)}, 
                     "comparison_map": {"nsamples_plot": 10, "nens": 3, "cmap_name": "PuOr", "cmap_range": (.5, 1.), 
                                        "levels": [0., 0.25, 0.5, 1., 1.5, 2.5, 5., 7.5, 10., 15., 20., 30., 50., 75.]}
                                       }

    eval_func(tot_prec_pred, tot_prec_tar, outdir=outdir, ens_dim="ensemble", eval_dict=eval_dict)
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation pipeline for precipitation forecasts.")
    parser.add_argument("--datadir", type=Path, help="Directory containing forecast data.")
    parser.add_argument("--outdir", type=Path, help="Directory to store evaluation results.")
    parser.add_argument("--model_id", type=str, default="netcdf", help="W&B ID of AtmoRep evaluation run.")
    parser.add_argument("--ens", type=Union[str, int], default="all", help="Type of ensemble to evaluate. Choose between 'mean', 'all', <int>.")
    parser.add_argument("--quick_evaluate", action="store_true", help="Quick evaluation with reduced number of samples.")
    parser.add_argument("--nforecasts_quick", type=int, default=10, help="Number of forecasts for quick evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator.")

    args = parser.parse_args()
    main(args)