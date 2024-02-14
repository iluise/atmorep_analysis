####################################################################################################
#
#  Copyright (C) 2023
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
#
#  description : Helper containing all utils for the training_analysis.py code
#                a.k.a. plot the training diagnostic routine.
#
#  license     : MIT licence
#
####################################################################################################

import zarr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import calendar
import random
import os

from utils.plotter import Plotter
from utils.read_atmorep_data import HandleAtmoRepData

def inspect_token_variables(data, name, plotter, label = "_"):
    print("Info:: Plotting along local time-step dimension: e.g. t=0,1,2..")
    plotter.plot_along_dim(data, name = f"{name}_vs_time_step", dim = "t", label = label)
    print("Info:: Plotting along local latitude dimension: e.g. y = 0,..,8")
    plotter.plot_along_dim(data, name = f"{name}_vs_local_lat", dim = "y", label = label)
    print("Info:: Plotting along local longitude dimension e.g. x = 0,..,8")
    plotter.plot_along_dim(data, name = f"{name}_vs_local_lon", dim = "x", label = label)

def inspect_geo_variables(data, name, plotter):
    print("Info:: plot latitude distribution")
    plotter.plot([da.lat for da in data], name = name+"_latitude", xlabel = "latitude")

    print("Info:: plot longitude distribution")
    plotter.plot([da.lon for da in data], name = name+"_longitude", xlabel = "longitude")

    print("Info:: plot time distribution")
    plotter.hist([da.datetime.dt.month for da in data], name = name+"_datetime", bins = 12, xlabel = "month", range = [1, 12], log_yscale = True)
      
def profile_geo_variables(data, name, plotter):
        print("Info:: plot latitude profile")
        plotter.profile(data, name = name+"_latitude", groupby_dim = "lat", mean_dim = ["t", "x"])
        print("Info:: plot longitude profile")
        plotter.profile(data, name = name+"_longitude", groupby_dim = "lon", mean_dim = ["t", "y"])

def compare_target_pred(da_pred, da_target, plotter):
    print("Info:: Plotting Target vs Predictions")
    plotter.plot({ "preds": da_pred, "target": da_target}, name = "target_vs_preds", log_yscale = True)

    inspect_geo_variables(da_pred, name = "pred", plotter = plotter)

    #target vs predictions plotted in loop over: t=0,1,2 lat=0,..8 lon=0,..8
    inspect_token_variables({"preds": da_pred, "target": da_target}, name = "target_preds", plotter = plotter)

    #plot 2D distributions
    #plotter.mapshow([da.sel(t=0) for da in da_pred], name = "pred", show_contours = True)
    #plotter.mapshow([da.sel(t=0) for da in da_target], name = "target", show_contours = True)

def visualize_predictions(da_pred, da_target, scores_all, cases, plotter):

    print("Info:: Visualize predictions")
    cmap_custom = plotter.CustomPalette()

    for case, index_list in cases.items():
        print("Start case: "+ case)

        subdir_out = plotter.out_dir+"/"+case+"/"
        if not os.path.exists(subdir_out):
            os.makedirs(subdir_out)

        samples, tokens = index_list[0], index_list[1]
        for sidx, sample in enumerate(samples):
            for rank, iidx in enumerate(tokens[sidx]):
                for time in [0]: #range(3): #loop over time
                    pred = da_pred[sample].isel(t = time, itoken = iidx)
                    targ = da_target[sample].isel(t = time, itoken = iidx)
                    vmin = np.min(np.minimum(pred, targ))
                    vmax = np.max(np.maximum(pred, targ))

                    fig, ax_temp = plt.subplots(figsize=(6, 6))
                    gs = fig.add_gridspec(2, 2)
                    ax_temp.remove()
                    ax = []
                    ax.append(fig.add_subplot(gs[0,0]))
                    ax.append(fig.add_subplot(gs[0,1]))
                    ax.append(fig.add_subplot(gs[1,:]))

                    im1 = ax[0].imshow(pred, cmap=cmap_custom, vmin=vmin, vmax=vmax)
                    ax[0].set_title('Prediction', color='dimgray')
                    ax[0].set_xticks([]), ax[0].set_yticks([])

                    im2 = ax[1].imshow(targ, cmap=cmap_custom, vmin=vmin, vmax=vmax)
                    ax[1].set_title('Target', color='dimgray')
                    ax[1].set_xticks([]), ax[1].set_yticks([])
                    ax[1].set_xticks([]), ax[1].set_yticks([])

                    y0 = targ.values.mean()
                    ax[2].axhline(y=y0, color='darkgray', linestyle='-', linewidth=0.5)

                    ax[2].plot(targ.values.flatten(), label = "Reference" , color = 'royalblue', linewidth = 1.5)
                    ax[2].plot(pred.values.flatten(), label = "Prediction", color='red', linewidth = 1.5)

                    ax[2].set_ylabel(plotter.field)
                    ax[2].set_xlabel("local bin")
                    ax[2].legend(frameon=False)
                    plt.tight_layout()

                    plt.savefig( subdir_out +'/plot_{}_{}_sample{}_idx{}_time{}_rank{}.png'.format( plotter.field, plotter.model_id, sample, iidx.values, time, rank))
                    plt.close('all') #clear matplotlib


def analyse_ensemble(da_ens, da_target, plotter, l2_scores = None):
    #plot ensemble vs target
    print("Info:: Plot Ensemble vs Target")
    log_ds = {f"_ensemble{i}": [da.isel(ensemble = i) for da in da_ens] for i in range (1, len(da_ens[0].ensemble.values))}
    log_ds["ensemble"] = [da.isel(ensemble = 0) for da in da_ens] #change label only for the first entry
    log_ds["target"] = da_target
    plotter.plot(log_ds, name = "ens_vs_target", log_yscale = True)
    del log_ds

    print("Info:: Plot the Ensemble standard deviation")
    #standard deviation of the ensemble plotted in loop over: t=0,1,2 lat=0,..8 lon=0,..8
    inspect_token_variables([da.std(dim = "ensemble") for da in da_ens], label = "ens", name = "ens_std_dev", plotter = plotter)
    profile_geo_variables([da.std(dim = "ensemble") for da in da_ens], name = "ens_std_dev", plotter = plotter)

    #print("Info:: plot 2D map")
    #average 2D map of the std deviation
    #plotter.mapshow([da.isel(t=0).std(dim = "ensemble") for da in da_ens], name = "ens_std_dev", show_contours = True)

    #scatter plot L2 vs std
    if l2_scores:
        print("Info:: plot scatter plot std dev vs L2 error")
        plotter.scatter_plot([da.std(dim = "ensemble") for da in da_ens], l2_scores, name = "svd_dev_vs_l2_err", xlabel = 'standard deviation', ylabel = 'L2 error')
    else:
        print("Warning:: l2 error not computed. Skipping the scatter plot.")
