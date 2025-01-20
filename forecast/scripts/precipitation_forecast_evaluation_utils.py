# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC), European Centre for Medium-Range Weather Forecasts (ECMWF), 
#                              European Organization for Nuclear Research (CERN) - IT
#
# SPDX-License-Identifier: MIT

__authors__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2025-01-20"
__update__ = "2025-01-20"

"""
Methods used in the evaluation pipeline for precipitation forecasts.
"""

# import packages
import sys
sys.path.append("../")
from pathlib import Path
from typing import Union, List, Dict
import numpy as np
import pandas as pd
import xarray as xr

from utils.plotting import plot_histogram, mapplot_comparison_det, mapplot_comparison_ens, plot_rank_histogram, plot_metric_line
from utils.metrics import Scores



from typing import Union, List, Dict

str_or_path = Union[str, Path]

def eval_deterministic_forecast(da_fcst: xr.DataArray, da_obs: xr.DataArray, outdir: str_or_path, model_name: str = "AtmoRep", ens_dim: str= "ens",
                                eval_dict: Dict = {"scores": {"thresh_ets": [0.1, .5, 1.]},
                                                   "histogram": {"legend_labels": ["AtmoRep", "IMERG"],
                                                                 "bins": [0., .01, 0.1, 0.5, 1., 2., 5., 10., 15., 20], "figsize": (12, 8)}, 
                                                    "comparison_map": {"nsamples_plot": 10, "cmap_name": "PuOr", "cmap_range": (.5, 1.), 
                                                                       "levels": [0., 0.25, 0.5, 1., 1.5, 2.5, 5., 7.5, 10., 15., 20., 30., 50., 75.]}
                                                   }):
    """
    Perform evaluation of a determinstic precipitation forecast which involves the following:
    * basic scores: RMSE, MAE, ETS
    * histograms
    * comparison plot
    Can be used for evaluating the ensemble mean or specific ensemble members.
    :param da_fcst: Data array providing deterministic forecast data
    :param da_fcst: Data array providing the ground truth data/observation
    :param outdir: Base directory to store evaluation plots and data
    :param model_name: name of model that is evaluated
    :param ens_dim: name of ensemble dimension (only relevant of individual ensemble member should be evaluated)
    :param eval_dict: Nested dictionary with sub-directories for "scores", "histogram", "comparison_map"
    """
    outdir = Path(outdir) if not isinstance(outdir, Path) else outdir
    
    nforecasts = len(da_fcst["init_time"])
    if ens_dim not in da_fcst.dims:
        fname_suffix = "mean"
        title_model = f"{model_name} mean"
    else:
        ens = da_fcst[ens_dim].values
        fname_suffix =  f"ens{ens:d}"
        title_model = f"{model_name} (ens_mem= {ens:d})"

    ### Evaluation in terms of basic scores  
    # To-Do:
    # * create map-plots of metrics
    score_dict = eval_dict["scores"] 
    # get score enigine without spatial averaging  
    score_engine = Scores(da_fcst, da_obs, avg_dims=["init_time"])

    # calculate scores
    rmse = score_engine("rmse")
    print(f"Lead-time averaged RMSE over {nforecasts:d} forecasts: {rmse.mean():.4f} mm/h.")
    plot_metric_line(rmse.mean(dim=["lat", "lon"]), model_name, metric = {"RMSE": "mm/h"}, value_range = score_dict.pop("val_range", [.0, .1]),
                     plt_fname=outdir.joinpath(f"plot_rmse_{model_name.lower()}_precip.png"))

    mae = score_engine("mae")
    print(f"Lead-time averaged MAE over {nforecasts:d} forecasts: {mae.mean():.4f} mm/h.")
    plot_metric_line(mae.mean(dim=["lat", "lon"]), model_name, metric = {"MAE": "mm/h"}, value_range = score_dict.pop("val_range", [.0, .1]),
                     plt_fname=outdir.joinpath(f"plot_mae_{model_name.lower()}_precip.png"))

    ets_all = []
    ets_name = []
    for thresh in eval_dict["scores"]["thresh_ets"]:
        ets = score_engine("ets", thresh=thresh)
        print(f"Lead-time averaged ETS with threshold {thresh:.1f} mm/h over {nforecasts:d} forecasts: {ets.mean():.4f}")
        plot_metric_line(ets.mean(dim=["lat", "lon"]), model_name, metric = {"ETS": "mm/h"}, value_range = score_dict.pop("val_range", [.0, 1.]),
                         plt_fname=outdir.joinpath(f"plot_ets_t{thresh:.2f}_{model_name.lower()}_precip.png"))
        
        ets_all.append(ets)
        ets_name.append(f"ets_t{thresh:.2f}")

    # save score to netCDF-file
    score_dict = {"rmse": rmse, "mae": mae, **dict(zip(ets_name, ets_all))}
    ds_scores = xr.Dataset(score_dict)
    fname_scores = outdir.joinpath(f"scores_imerg_{model_name.lower()}_{fname_suffix}.nc")
    print(f"Save scores to '{fname_scores}'.")
    ds_scores.to_netcdf(outdir.joinpath(f"scores_imerg_{model_name.lower()}_{fname_suffix}.nc"))

    ### Produce histograms
    hist_kwargs = eval_dict["histogram"]
    plot_histogram(da_fcst, da_obs, outdir.joinpath(f"plot_histogram_imerg_{model_name.lower()}_{fname_suffix}_precip.png"), lshow=False,
                  **hist_kwargs)

    ### Create comparison plots
    nfcst_plt = min(eval_dict["comparison_map"].pop("nforecasts_plot", 10), nforecasts)
    plt_config = eval_dict["comparison_map"].copy()
    for i in range(nfcst_plt):
        da_fcst_init = da_fcst.isel({"init_time": i})
        init_now = convert_to_datetime(da_fcst_init['init_time'].values)
        for lead_time in da_fcst_init['lead_time'].values:
            da_fcst_now = da_fcst_init.sel({'lead_time': lead_time})
            plt_config["sup_title"] = f"{init_now.strftime('%Y/%m/%d %H:00')} UTC +{int(lead_time):03d} h"
            plt_config["titles"] = ["IMERG"] + [title_model]
            plt_fname = outdir.joinpath(f"plot_imerg_{model_name.lower()}_{fname_suffix}_precip_"+
                                        f"{init_now.strftime('%Y%m%d-%H00')}+{int(lead_time):03d}.png")
        
            mapplot_comparison_det(da_obs.isel({"init_time": i,}).sel({"lead_time": lead_time}), da_fcst_now,
                                   plt_fname, lshow=False, **plt_config.copy())
            

def eval_probablistic_forecast(da_fcst: xr.DataArray, da_obs: xr.DataArray, outdir: str_or_path, model_name: str = "AtmoRep", ens_dim : str ="ens",
                               eval_dict : Dict = {"scores": {}, 
                                                   "histogram": {"legend_labels": ["AtmoRep", "IMERG"],
                                                                 "bins": np.array([0., .01, 0.1, 0.5, 1., 2., 5., 10., 15., 20]), "figsize": (12, 8)}, 
                                                   "comparison_map": {"nforecasts_plot": 10, "nens": 3, "cmap_name": "PuOr", "cmap_range": (.5, 1.), 
                                                                      "levels": [0., 0.25, 0.5, 1., 1.5, 2.5, 5., 7.5, 10., 15., 20., 30., 50., 75.]}
                                                   }):
    """
    Perform evaluation of a probablistic precipitation forecast which involves the following:
    * basic scores: CRPS, rank histogram
    * histograms
    * comparison plot
    Can be used for evaluating the ensemble mean or specific ensemble members.
    :param da_fcst: Data array providing deterministic downscaling data
    :param da_obs: Data array providing the ground truth data/observation
    :param outdir: Base directory to store evaluation plots and data
    :param model_name: name of model that is evaluated
    :param ens_dim: name of ensemble dimension 
    :param eval_dict: Nested dictionary with sub-directories for "scores", "histogram", "comparison_map"
    """    
    outdir = Path(outdir) if not isinstance(outdir, Path) else outdir

    fname_suffix = "ensemble" 
    
    nforecasts = len(da_fcst["init_time"])
    ### Evaluation in terms of basic scores
    score_dict = eval_dict["scores"]  
    score_engine = Scores(da_fcst, da_obs, avg_dims=["init_time", "lat", "lon"], ens_dim=ens_dim)

    crps = score_engine("crps")
    print(f"Lead-time averaged CRPS over {nforecasts:d} forecasts: {crps.mean():.4f} mm/h.")
    
    plot_metric_line(crps, model_name, metric = {"crps": "mm/h"}, value_range = score_dict.pop("val_range", [.0, .1]),
                     plt_fname=outdir.joinpath(f"plot_crps_{model_name.lower()}_precip.png"))

    # Perform rank histogram calculation over all dimensions
    score_engine.avg_dims = ["init_time", "lat", "lon"]
    rank_norm = score_engine("rank_histogram")
    # Plot normalized rank histogram for each lead time separately   
    for lead_time in rank_norm["lead_time"].values: 
        plot_rank_histogram(rank_norm.sel({"lead_time": lead_time}),
                            outdir.joinpath(f"rank_histogram_{model_name.lower()}_{fname_suffix}_leadtime+{int(lead_time):03d}h"), **score_dict)

    ### Produce histogram
    hist_kwargs = eval_dict["histogram"]
    plot_histogram(da_fcst, da_obs, outdir.joinpath(f"plot_histogram_imerg_{model_name}_{fname_suffix}_precip.png"), 
                   ens_dim = ens_dim, lshow=False, **hist_kwargs)

    ### Create comparison plots
    nfcst_plt = min(eval_dict["comparison_map"].pop("nforecasts_plot", 10), nforecasts)
    plt_config = eval_dict["comparison_map"].copy()
    for i in range(nfcst_plt):
        da_fcst_init = da_fcst.isel({"init_time": i})
        init_now = convert_to_datetime(da_fcst_init['init_time'].values)
        for lead_time in da_fcst_init['lead_time'].values:
            da_fcst_now = da_fcst_init.sel({'lead_time': lead_time})
            plt_config["sup_title"] = f"{init_now.strftime('%Y/%m/%d %H:00')} UTC +{int(lead_time):03d} h"
            plt_config["titles"] = ["IMERG"] + [f"{model_name} ens={ens:d}" for ens in range(plt_config["nens"])]
            plt_fname = outdir.joinpath(f"plot_imerg_{model_name.lower()}_{fname_suffix}_precip_"+
                                        f"{init_now.strftime('%Y%m%d-%H00')}+{int(lead_time):03d}.png")
        
            mapplot_comparison_ens(da_obs.isel({"init_time": i,}).sel({"lead_time": lead_time}), da_fcst_now, plt_fname,
                                   ens_name=ens_dim, lshow=False, ens_dim=ens_dim, **plt_config)

def convert_to_datetime(time_obj):
    """
    Convert numpy datetime or tuple of numpy datetime and lead time to a pandas datetime-object.
    """
    if isinstance(time_obj, (tuple, list, np.ndarray)):
        try:
            time_obj = list(time_obj)
        except:      # if the time-data is from a xarray MultiIndex, time_obj is a 0-dimensional array
            time_obj = list(time_obj.item())
        t_dt = pd.to_datetime(time_obj[0]) + pd.Timedelta(time_obj[1])
    else:
        t_dt = pd.to_datetime(time_obj)

    return t_dt
