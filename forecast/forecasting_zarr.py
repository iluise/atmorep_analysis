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

import pandas as pd
import datetime
import json
from utils.metrics import Scores, calc_scores_item
from utils.plotter import Plotter
from utils.read_atmorep_data import HandleAtmoRepData
from utils.utils import get_pl_level, grib_index
#from dask.distributed import Client

models_id  = {
#     '1arfi1oi': '12 step -> 6h - no fine-tuning - Paper',
#     '5yyn8vu6': '12 step -> 6h - Paper - epoch 45', #36h
#     'lpfi1lul': '12 step -> 24h new-training - epoch 7',  #36h

#     'c96xrbip': '1 step -> 6h new-training - epoch 10', #9h
#     'l0cvsqhm': '1 step -> 6h new-training - epoch 15', #9h
#     'bp7vm29f': '2 step -> 6h new-training - epoch 10', #12h
#     'irk7o1op': '4 step -> 6h new-training - epoch 10'  #18h
#     'su4kfzxv': '4 step -> 12h new-training - epoch 5'  #18h        
    #'s23kteu6' : '6 days',
    #velocity_u
    # 'rd4c6tbu' : '7 days - CRPS',
    #'7w7rh21c' : '7 days weight. MSE + stats',
    #'dtqm59uy': '7 days MSE + stats'
    #temperature
    #'x7n3hnpv' :'7 days - CRPS', 
    # 'vuk4jkmf': '7 days weight. MSE + stats',
    # 'acsvut65' : '7 days - MSE ensemble + stats'

    #specific humidity
   
    #'199j7u50': '3x9x9 global norm',
    'd018p5pk': '3x18x18 global norm',
    'lld5w50x': '3x9x9 old config',

}
plot_pangu = False
plot_ensemble = True
experiment = "test" #"pred24h_atmorep_only"
input_dir = "./results/"
out_dir   = "./figures/forecast/"
fields    = ["specific_humidity"] #["velocity_u"] #["temperature", "specific_humidity", "velocity_u", "velocity_v"]
#fields    = ["temperature"] #["velocity_u"] #["temperature", "specific_humidity", "velocity_u", "velocity_v"]
metrics   = ["ssr"] #"spread"] #, "acc", "spread"] #"rmse", "acc"]
levels    = [123] #, 123, 137, 105, 96]
pl_levels = {137 : 1000, 123 : 925, 114 : 850, 105 : 700, 96: 500}
#basedir = "/p/scratch/atmo-rep/data/"
fcst_hours = 6


if plot_pangu:
        pangu = xr.open_dataset("/p/scratch/atmo-rep/data/Pangu-Weather/output_panguweather_6h_forecast_202102.nc", engine= "netcdf4").sel(isobaricInhPa=[pl for pl in pl_levels.values()]).sortby('time')
        print("PanguWeather - opened")
        pangu_plots = []
        pangu_levels = [pl_levels[lvl] for lvl in levels]

        for fidx, field in enumerate(fields):
                ds_pangu = pangu.sel(isobaricInhPa=pangu_levels)
                datetimes_pangu  = ds_pangu.time
                pangu_fcst_hours = np.sort(np.unique(ds_pangu.step))
	        #breakpoint()
                datetimes_era5 = [dt + pd.DateOffset(hours=int(i)) for i in range(1, len(pangu_fcst_hours)+1) for dt in datetimes_pangu.values]
                era5_pl_name = [f"/p/fastdata/slmet/slmet111/met_data/ecmwf/era5_reduced_level/pl_levels/{pl_levels[lvl]}/{field}/reanalysis_{field}_y2021_m02_pl{pl_levels[lvl]}.grib" for lvl in levels]
                era5_pl  = xr.concat([xr.open_dataset(fname, engine = "cfgrib", backend_kwargs={"indexpath": ''}).sel(time = datetimes_era5)[grib_index[field]].sortby('time') for fname in era5_pl_name], dim = 'isobaricInhPa').transpose('time', 'isobaricInhPa', 'latitude', 'longitude')

                ds_era5 = xr.Dataset( coords={
	                'time'          : datetimes_pangu,
	                'step'          : pangu_fcst_hours, #np.sort(np.unique(ds_pangu.step)),
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
            da_target  = ar_data.read_data(field, "target", ml = levels)
            da_ens     = ar_data.read_data(field, "ens"   , ml = levels) if (metrics.count('ssr') > 0 or metrics.count('spread') > 0)  else None
            da_pred    = ar_data.read_data(field, "pred"  , ml = levels)
                    
#           da_source  = ar_data.read_data(field, "source", ml = level).sortby('datetime').sel(ml = level)
            
            clim_mean = xr.open_zarr("climatological_mean.zarr")[field].sel(ml = levels)
            options = {"clim_mean":  clim_mean}

            atmorep_scores = np.array(calc_scores_item(da_pred, da_target, da_ens, metrics, options, ["lat", "lon"]))
            atmorep_scores = atmorep_scores.reshape( *atmorep_scores.shape[:2], -1, fcst_hours).mean(axis = -2)
            plots[mod_idx,fidx] = atmorep_scores[:]

######### debug ########
            # print("da_pred.values", model_id, da_pred.values.shape)
            # temp =  da_pred.sel(ml = 114).values.squeeze()            
            # print("temp.values", model_id, temp.shape)
           
            # for t in range(temp.shape[0]):
            #     fig = plt.figure()
            #     plt.imshow(temp[t], cmap = 'PuBuGn', origin= 'lower')
            #     # breakpoint()
            #     plt.title(da_pred['datetime'].values[t])
            #     fig.savefig(f"{model_id}/map_target_{field}_ml114_time{t}.png")
            #     plt.close()

#now plot
for fidx, field in enumerate(fields):    
    for lidx, level in enumerate(levels):
        for midx, metric in enumerate(metrics):
            fig1, ax1 = plt.subplots()
            #if plot_pangu:
            #    ax1.plot( list(range(1, len(pangu_fcst_hours)+1)), pangu_plots[fidx][midx, lidx], label = "PanguWeather")
            for mod_idx, (model_id, label) in enumerate(models_id.items()):
                print(plots[mod_idx][fidx][midx, lidx])
                linestyle = "dashed" if "non fine-tuned" in label else "solid"
                ax1.plot( list(range(1, fcst_hours+1)), plots[mod_idx][fidx][midx, lidx], label = label, linestyle = linestyle)
            if plot_pangu:
                ax1.plot( list(range(1, len(pangu_fcst_hours)+1)), pangu_plots[fidx][midx, lidx], label = "PanguWeather", color = "black")
            plt.subplots_adjust(right=0.7)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon= False)
            print(f"test_{metric}_{field}_{level}_{experiment}.png")
            fig1.savefig(f"test_{metric}_{field}_{level}_{experiment}.png",  bbox_inches="tight")
        
