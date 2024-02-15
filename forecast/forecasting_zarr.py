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
import pdb
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
from utils.utils import get_pl_level, grib_index

models_id  = {
     #'1arfi1oi': '10 step -> 6h - no fine-tuning - Paper',
     '5yyn8vu6': '10 step -> 6h - Paper',

     'bp7vm29f': '2 step -> 6h new-training - epoch 4', #12h
     'c96xrbip': '1 step -> 6h new-training - epoch 4', #9h
     's4mvp86e': '4 step -> 6h new-training - epoch 1'  #18h
              }

experiment = "pred6h"
input_dir = "/p/scratch/atmo-rep/results/"
out_dir   = "./figures/forecast/"
fields    = ["temperature", "specific_humidity", "velocity_u", "velocity_v"]
#fields    = ["total_precip"]
metrics   = ["rmse"] #, "acc"]
levels    = [137, 123]
pl_levels = {137 : 1000, 123 : 925, 114 : 850, 105 : 700, 96: 500}
basedir = "/p/scratch/atmo-rep/data/"
fcst_hours = 6


pangu = xr.open_dataset("/p/scratch/atmo-rep/data/Pangu-Weather/output_panguweather_6h_forecast_202102.nc", engine= "netcdf4").sel(isobaricInhPa=[pl for pl in pl_levels.values()]).sortby('time')
print("PanguWeather - opened")
pangu_plots = []
pangu_levels = [pl_levels[lvl] for lvl in levels]

for fidx, field in enumerate(fields):
    ds_pangu = pangu.sel(isobaricInhPa=pangu_levels)
    datetimes_pangu = ds_pangu.time
    #breakpoint()
    datetimes_era5 = [dt + pd.DateOffset(hours=int(i)) for i in range(1, fcst_hours+1) for dt in datetimes_pangu.values]
    era5_pl_name = [f"/p/fastdata/slmet/slmet111/met_data/ecmwf/era5_reduced_level/pl_levels/{pl_levels[lvl]}/{field}/reanalysis_{field}_y2021_m02_pl{pl_levels[lvl]}.grib" for lvl in levels]

    era5_pl  = xr.concat([xr.open_dataset(fname, engine = "cfgrib", backend_kwargs={"indexpath": ''}).sel(time = datetimes_era5)[grib_index[field]].sortby('time') for fname in era5_pl_name], dim = 'isobaricInhPa').transpose('time', 'isobaricInhPa', 'latitude', 'longitude')
    
    ds_era5 = xr.Dataset( coords={
        'time'          : datetimes_pangu,
        'step'          : np.sort(np.unique(ds_pangu.step)),
        'isobaricInhPa' : pangu_levels,
        'latitude'      : era5_pl.latitude.values,
        'longitude'     : era5_pl.longitude.values
    } )

    ds_era5[grib_index[field]] = (['time', 'step', 'isobaricInhPa', 'latitude', 'longitude'], era5_pl.values.reshape(ds_pangu[grib_index[field]].shape))
    pangu_scores = np.array(calc_scores_item(ds_pangu[grib_index[field]], ds_era5[grib_index[field]], metrics, ["latitude", "longitude", "time"])).transpose(0, 2, 1)
    print("shape", pangu_scores.shape)
    pangu_plots.append(pangu_scores)


####### atmorep #######
plots = np.zeros([len(models_id), len(fields), len(metrics),  len(levels), fcst_hours])
for mod_idx, (model_id, label) in enumerate(models_id.items()):
    ar_data = HandleAtmoRepData(model_id, input_dir)
    for fidx, field in enumerate(fields):
            da_target  = ar_data.read_data(field, "target", ml = levels).sortby('datetime').sel(ml = levels)       
            da_pred    = ar_data.read_data(field, "pred"  , ml = levels).sortby('datetime').sel(ml = levels)
#           da_source  = ar_data.read_data(field, "source", ml = level).sortby('datetime').sel(ml = level)

            atmorep_scores = np.array(calc_scores_item(da_pred, da_target, metrics, ["lat", "lon"]))
            print(atmorep_scores.shape)
            atmorep_scores = atmorep_scores.reshape( *atmorep_scores.shape[:2], -1, fcst_hours).mean(axis = -2)
            print(atmorep_scores.shape)
            plots[mod_idx,fidx] = atmorep_scores[:]

######### debug ########
#            print("da_pred.values", model_id, da_pred.values.shape)
#            temp =  da_target.values.squeeze()            
#            print("temp.values", model_id, temp.shape)
#            for t in range(temp.shape[0]):
#                fig = plt.figure()
#                plt.imshow(temp[t], cmap = 'PuBuGn', origin= 'lower')
#                fig.savefig(f"{model_id}/map_target_{field}_ml{level}_time{t}.png")
#                plt.close()

#now plot
for fidx, field in enumerate(fields):    
    for lidx, level in enumerate(levels):
        for midx, metric in enumerate(metrics):
            fig1, ax1 = plt.subplots()
            ax1.plot( list(range(1, fcst_hours+1)), pangu_plots[fidx][midx, lidx], label = "PanguWeather")
            for mod_idx, (model_id, label) in enumerate(models_id.items()):
              print(plots[mod_idx][fidx][midx, lidx])
              linestyle = "dashed" if "non fine-tuned" in label else "solid"
              ax1.plot( list(range(1, fcst_hours+1)), plots[mod_idx][fidx][midx, lidx], label = label, linestyle = linestyle)
            plt.subplots_adjust(right=0.7)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon= False)
            fig1.savefig(f"test_{metric}_{field}_{level}_{experiment}.png",  bbox_inches="tight")
        
