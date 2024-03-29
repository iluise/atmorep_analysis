#
#  Copyright (C) 2023
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
#
#  description : Code to plot the training diagnostic routine.
#                It monitors target, preds, source and the ensemble.
#
#  license     : MIT licence
#
####################################################################################################

import zarr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import calendar
import random
import os

from utils.metrics import Scores, calc_scores_item
from utils.plotter import Plotter
from utils.read_atmorep_data import HandleAtmoRepData
from trainings.training_utils import *

model_id  = 'idmi4tsq5d' 
input_dir = "/p/scratch/atmo-rep/results/"
out_dir   = "./figures/" 
field     = "vorticity"
metrics   = ["l2"]
levels    = [137, 123] 
load_source = False


ar_data = HandleAtmoRepData(model_id, input_dir)
mean_scores = {metric : [] for metric in metrics}
for level in levels:
    # list of xarray DataArrays is returned where each element corresponds to a sample (patch)
    da_pred    = ar_data.read_data(field, "pred"  , ml = level)
    da_target  = ar_data.read_data(field, "target", ml = level)
    da_source  = ar_data.read_data(field, "source", ml = level) if load_source else None
    
    #instantiate plotting class
    plotter = Plotter(field, model_id, out_dir, level)
    
    #target vs preds
    compare_target_pred(da_pred, da_target, plotter)

    ##plot source - optional. it returns zero if source is None
    plotter.plot(da_source, name = "source")
    ######################################
    
    #compute errors:
    print(f"Info:: Computing Metrics: {metrics}")
    scores_all = [calc_scores_item(pred, target, metrics) for pred, target in zip(da_pred, da_target)]
    
    # inspect variables for each computed metric
    for midx, metric in enumerate(metrics):
        inspect_token_variables([s[midx] for s in scores_all], label = metric, name = metric, plotter = plotter)
        
        profile_geo_variables([s[midx] for s in scores_all], name = metric, plotter = plotter)

        #plot 2D map, only timestep = 0
        plotter.mapshow( [s[midx].sel(t=0) for s in scores_all] , name = metric, show_contours = True)

        #store mean value: per-level statistics
        mean_scores[metric].append(np.mean([s[midx].mean() for s in scores_all]))

        #sort errors from max to min
        #visualize single cases: get samples with max and min avg error
        samplOrd = np.flip( np.argsort([s[midx].mean() for s in scores_all]))
        
        idxsOrd = []
        for sidx in samplOrd:  #sort max to min tokens within each sample. save in a list with same simensions as sample.
            toknOrd = np.flip(np.argsort(scores_all[sidx][midx].mean(dim = ["x", "y", "t"])))
            idxsOrd.append(toknOrd)

        N = 10
        rndm_idx = random.choices(samplOrd,k=50)
        cases = {
            "abs_err_single_sample_max" : ([samplOrd[0]], [idxsOrd[0][:N]]),      #first 10 tokens with highest abs error from the same sample with highest avg error  
            "abs_err_single_sample_min" : ([samplOrd[0]], [idxsOrd[0][-N:]]),     #first 10 tokens with lowest abs error from the sample with lowest avg error
            "abs_err_multi_samples_max" : (samplOrd[:N],  [[idxsOrd[i][0]] for i in range(N)]),  #token with highest abs error from the 10 samples with highest avg error
            "abs_err_multi_samples_min" : (samplOrd[-N:],  [[idxsOrd[-i][0]] for i in range(N, 0, -1)]),  #token with lowest abs err from the 10 samples with lowest avg error
            "rndm"        : ([ samplOrd[r] for r in rndm_idx], [[idxsOrd[r][0]] for r in rndm_idx])        #first tokens from random 50 samples
            #    "iidx"        : [idxsOrd[0]]                  #specific index e.g. 2*3456-1
        }
        
        visualize_predictions(da_pred, da_target, scores_all, cases, plotter)
        
    print("Info:: Start Ensemble Analysis")
    da_ens  = ar_data.read_data(field, "ens", ml = level)
    analyse_ensemble(da_ens, da_target, l2_scores = [s[metrics.index("l2")] for s in scores_all] if "l2" in metrics else None, plotter = plotter)

#plot as a function of the vertical level
for metric in metrics:
    print(mean_scores[metric])
    plotter.graph(mean_scores[metric], levels, name = f"{metric}_vs_level", xlabel = "level", ylabel = metric)







