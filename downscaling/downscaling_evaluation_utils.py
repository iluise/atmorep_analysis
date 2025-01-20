# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC), European Centre for Medium-Range Weather Forecasts (ECMWF), 
#                              European Organization for Nuclear Research (CERN) - IT
#
# SPDX-License-Identifier: MIT

__authors__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2024-12-16"
__update__ = "2025-01-20"

"""
Methods used in the evaluation pipeline of the downscaling application.
"""

# import packages
from pathlib import Path
from typing import Union, List, Dict
import numpy as np
import pandas as pd
import xarray as xr

from ..utils.plotting import plot_histogram, mapplot_comparison_det, mapplot_comparison_ens, plot_rank_histogram
from ..utils.metrics import Scores

str_or_path = Union[str, Path]

# main evaluation functions

def eval_deterministic_downscaling(da_fcst: xr.DataArray, da_obs: xr.DataArray, outdir: str_or_path, model_name: str = "Harris WGAN", ens_dim: str = "ens",
                                   eval_dict: Dict = {"scores": {"thresh_ets": [0.1, .5, 1.]},
                                                      "histogram": {"legend_labels": ["Harris WGAN", "IMERG"],
                                                                    "bins": [0., .01, 0.1, 0.5, 1., 2., 5., 10., 15., 20], "figsize": (12, 8)}, 
                                                      "comparison_map": {"nsamples_plot": 10, "cmap_name": "PuOr", "cmap_range": (.5, 1.), 
                                                                         "levels": [0., 0.25, 0.5, 1., 1.5, 2.5, 5., 7.5, 10., 15., 20., 30., 50., 75.]}
                                                     }):
    """
    Perform a deterministic downscaling evaluation which involves the following:
    * basic scores: RMSE, MAE, ETS
    * histograms
    * comparison plot
    Can be used for evaluating the ensemble mean or specific ensemble members.
    :param da_fcst: Data array providing deterministic forecasts
    :param da_obs: Data array providing the ground truth data/observation
    :param model_name: Name of the model to be evaluated
    :param ens_dim: name of ensemble dimension (only relevant of individual ensemble member should be evaluated)
    :param outdir: Base directory to store evaluation plots and data
    :param eval_dict: Nested dictionary with sub-dictionaries for scores, histogram, comparison_map
    """
    outdir = Path(outdir) if not isinstance(outdir, Path) else outdir
    
    nsamples = len(da_fcst["time"])

    if ens_dim in da_fcst.dims:
        ens = da_fcst[ens_dim].values
        fname_suffix = f"ens{ens:d}"
        title_model = f"{model_name} (ens_mem= {ens:d})"
    else:
        fname_suffix = f"mean"
        title_model = f"{model_name} mean"

    ### Evaluation in terms of basic scores  
    # To-Do:
    # * create map-plots of metrics
    
    # get score enigine without spatial averaging  
    score_engine = Scores(da_fcst, da_obs, avg_dims=["time"])

    # calculate scores
    rmse = score_engine("rmse")
    print(f"Domain-averaged RMSE over {nsamples:d} samples: {rmse.mean():.4f} mm/h.")

    mae = score_engine("mae")
    print(f"Domain-averaged MAE over {nsamples:d} samples: {mae.mean():.4f} mm/h.")

    ets_all = []
    ets_name = []
    for thresh in eval_dict["scores"]["thresh_ets"]:
        ets = score_engine("ets", thresh=thresh)
        print(f"Domain-averaged ETS with threshold {thresh:.1f} mm/h over {nsamples:d} samples: {ets.mean():.4f}")
        ets_all.append(ets)
        ets_name.append(f"ets_t{thresh:.2f}")

    # save score to netCDF-file
    score_dict = {"rmse": rmse, "mae": mae, **dict(zip(ets_name, ets_all))}
    ds_scores = xr.Dataset(score_dict)
    fname_scores = outdir.joinpath(f"scores_imerg_{model_name.lower()}_{fname_suffix}.nc")
    print(f"Save scores to '{fname_scores}'.")
    ds_scores.to_netcdf(fname_scores)

    ### Produce histograms
    hist_kwargs = eval_dict["histogram"]
    plot_histogram(da_fcst, da_obs, outdir.joinpath(f"plot_histogram_imerg_{model_name.lower()}_{fname_suffix}_precip.png"), lshow=True, 
                  **hist_kwargs)

    ### Create comparison plots
    nsamples_plt = min(eval_dict["comparison_map"].pop("nsamples_plot", 0), nsamples)
    plt_config = eval_dict["comparison_map"].copy()
    for i in range(nsamples_plt):#range(len(tidx)):
        da_fcst_now = da_fcst.isel({"time": i})
        time_str = pd.to_datetime(da_fcst_now['time'].values).strftime('%Y%m%d-%H00')
        plt_config["titles"] = ["IMERG", title_model]
        plt_fname = outdir.joinpath(f"plot_imerg_{model_name.lower()}_{fname_suffix}_precip_{time_str}.png")
        
        mapplot_comparison_det(da_obs.isel({"time": i}), da_fcst_now, plt_fname, lshow=True, **plt_config.copy())


def eval_probablistic_downscaling(da_fcst: xr.DataArray, da_obs: xr.DataArray, outdir: str_or_path, model_name: str = "Harris WGAN", ens_dim: str = "ens",
                                  eval_dict : Dict = {"scores": {}, 
                                                      "histogram": {"legend_labels": ["Harris WGAN", "IMERG"],
                                                                    "bins": np.array([0., .01, 0.1, 0.5, 1., 2., 5., 10., 15., 20]), "figsize": (12, 8)}, 
                                                      "comparison_map": {"nsamples_plot": 10, "nens": 3, "cmap_name": "PuOr", "cmap_range": (.5, 1.), 
                                                                         "levels": [0., 0.25, 0.5, 1., 1.5, 2.5, 5., 7.5, 10., 15., 20., 30., 50., 75.]}
                                                     }):
    """
    Perform a probablistic downscaling evaluation which involves the following:
    * basic scores: CRPS, rank histogram
    * histograms
    * comparison plot
    Can be used for evaluating the ensemble mean or specific ensemble members.
    :param da_fcst: Data array providing deterministic forecasts
    :param da_obs: Data array providing the ground truth data/observation
    :param outdir: Base directory to store evaluation plots and data
    :param model_name: Name of the model
    :param ens_dim: name of ensemble dimension
    :param eval_dict: Nested dictionary with sub-dictionaries for scores, histogram, comparison_map
    """
    outdir = Path(outdir) if not isinstance(outdir, Path) else outdir

    fname_suffix = "ensemble" 
    
    nsamples = len(da_fcst["time"])
    ### Evaluation in terms of basic scores
    score_dict = eval_dict["scores"]  
    score_engine = Scores(da_fcst, da_obs, avg_dims=["time"])

    crps = score_engine("crps")
    print(f"Domain-averaged CRPS over {nsamples:d} samples: {crps.mean().values:.4f} mm/h.")

    # Perform rank histogram calculation over all dimensions
    score_engine.avg_dims = ["time", "lat", "lon"]
    rank_norm = score_engine("rank_histogram")
    
    # Plot normalized rank histogram    
    plot_rank_histogram(rank_norm, outdir.joinpath(f"rank_histogram_{model_name.lower()}_{fname_suffix}"), **score_dict)

    # save score to netCDF-file
    score_dict = {"crps": crps, "normalized_rank": rank_norm}
    ds_scores = xr.Dataset(score_dict)
    fname_scores = outdir.joinpath(f"scores_imerg_wgan_{fname_suffix}.nc")
    print(f"Save scores to '{fname_scores}'.")
    ds_scores.to_netcdf(outdir.joinpath(f"scores_imerg_{model_name.lower()}_{fname_suffix}.nc"))

    ### Produce histograms
    hist_kwargs = eval_dict["histogram"]
    plot_histogram(da_fcst, da_obs, outdir.joinpath(f"plot_histogram_imerg_{model_name.lower()}_{fname_suffix}_precip.png"), 
                   lshow=False, **hist_kwargs)

    ### Create comparison plots
    nsamples_plt = min(eval_dict["comparison_map"].pop("nsamples_plot", 10), nsamples)
    plt_config = eval_dict["comparison_map"].copy()
    nsamples_plt = 0
    for i in range(nsamples_plt):#range(len(tidx)):
        da_fcst_now = da_fcst.isel({"time": i})
        date_now = pd.to_datetime(da_fcst_now['time'].values)
        plt_config["sup_title"] = f"{date_now.strftime('%Y/%m/%d %H:00')} UTC"
        plt_config["titles"] = ["IMERG"] + [f"{model_name} ens={ens:d}" for ens in range(plt_config["nens"])]
        plt_fname = outdir.joinpath(f"plot_imerg_{model_name.lower()}_{fname_suffix}_precip_{date_now.strftime('%Y%m%d-%H00')}.png")
        
        mapplot_comparison_ens(da_obs.isel({"time": i}), da_fcst_now, plt_fname, lshow=False, **plt_config)