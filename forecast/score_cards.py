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
#  description : Code to create score cards. Not used in the paper. Just for R&D. 
#
#  license     : MIT licence
#                
####################################################################################################

import numpy as np
import xarray as xr
import properscoring as ps
import pandas as pd

from utils.utils import *
from utils.metrics import *
from utils.plotting import *
from datetime import datetime 

import netCDF4
import cfgrib

# plotting
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns # for data visualization
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 0.1
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.colors as colors
import itertools
from forecasting_utils import *

model_id  = '5eoirvtp' 

fields = ["velocity_u", "velocity_v","temperature",  "specific_humidity"]
field_name = ["U component", "V component",  "Temperature", "Specific Humidity"]

ml_levels = [137, 123, 114, 105, 96]
pl_levels = [1000, 925, 850, 700, 500]

year, month, day = 2018, 1, 2
out_dir = './'+model_id+'_'+str(year) + '_' +str(month)+ '_' + str(day) + "_nft_analysis1" #"_add_3h"
average = True

def compute_metrics_for_score_card():

  for field in fields: 
    fname_atrep = ["./netcdf/atmorep_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format(field, model_id, year, month, ml_level) for ml_level in ml_levels]
    fname_atrep_target = ["./netcdf/atmorep_target_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format( field, model_id, year, month, ml_level) for ml_level in ml_levels]
   
    #atmorep
    atrep_target = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep_target], dim = "vlevel") #.isel(time = np.arange(7,9))
    atrep = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep], dim = "vlevel") #.isel(time = np.arange(7,9))
    datetimes = [pd.to_datetime(x) for x in np.unique(atrep_target.time.values)]
    steps = np.sort(np.unique(atrep_target.step))
    print (datetimes)

    #era5
    fname_era5_ml =  ["/p/scratch/atmo-rep/data/era5/ml_levels/{}/{}/reanalysis_{}_y{}_m{:02d}_ml{}.grib".format(ml_level, field, field, year, month, ml_level) for ml_level in ml_levels]
    era5_ml  = xr.concat([xr.open_dataset(fname, engine = "cfgrib").sel(time = slice(datetimes[0], datetimes[-1]+ pd.DateOffset(hours=int(steps[-1])))) for fname in fname_era5_ml], 
                      dim = "vlevel").transpose('vlevel', 'time', 'latitude', 'longitude')[grib_index[field]].values                             
    print("Info:: ERA5 ML - Opened.")
    
    #IFS
    datetimes_hack = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond, d.tzinfo)+ pd.DateOffset(hours=1) for d in datetimes]
    fname_era5_ml_hack =  ["/p/scratch/atmo-rep/data/era5/ml_levels/{}/{}/reanalysis_{}_y{}_m{:02d}_ml{}.grib".format(ml_level, field, field, year, month, ml_level) for ml_level in ml_levels]
    era5_ml_hack  = xr.concat([xr.open_dataset(fname, engine = "cfgrib").sel(time = slice(datetimes_hack[0], datetimes_hack[-1]+ pd.DateOffset(hours=int(steps[-1])))) for fname in fname_era5_ml_hack], 
                      dim = "vlevel").transpose('vlevel', 'time', 'latitude', 'longitude')[grib_index[field]].values                             
    print("Info:: ERA5 ML - Opened.")

    fname_ifs   = ["/p/scratch/atmo-rep/data/ifs/ml_levels/{}/{}/reanalysis_{}_y{}_m{:02d}_ml{}.nc".format(ml_level, field, field, year, month, ml_level) for ml_level in ml_levels]
    ifs = xr.concat([xr.open_dataset(fname, engine = "netcdf4")  for fname in fname_ifs], dim = "vlevel")
    print(np.unique(ifs.time.values)) 
    ifs = xr.concat([xr.open_dataset(fname, engine = "netcdf4").sel(time = slice(datetimes_hack[0], datetimes_hack[-1]+ pd.DateOffset(hours=int(steps[-1])))) for fname in fname_ifs], 
                      dim = "vlevel").transpose('vlevel', 'time','lat', 'lon')[grib_index[field]].values                           
    print("Info:: IFS - Opened.")

    nsteps = len(steps)
    ntimes = len(datetimes)
    nlevels = len(ml_levels)

    atrep_target = atrep_target.transpose('time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values
    print("Info:: AtmoRep target - Opened.")
    atrep = atrep.transpose('time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values 
    print("Info:: AtmoRep - Opened.")
  
    print("Info:: atmorep     ", atrep.shape)
    print("Info:: nsteps ", nsteps)

    #Compute metrics:
    metrics_names = ["ACC", "RMSE"] #"NRMSE", "L2", "L1"] 
    shape = [len(metrics_names), ntimes , nlevels, nsteps]
    atrep_target_metrics = np.zeros(shape)
    ifs_metrics = np.zeros(shape)
    for m, metric in enumerate(metrics_names):
      for i, time in enumerate(datetimes):
        print("Running step:: ", time)
        atrep_target_metrics[m, i] = get_metrics(metric, field, atrep_target[i] , atrep[i], time, ml_levels) 
        ifs_metrics[m, i]   = get_metrics(metric, field, era5_ml_hack[:, i:i+nsteps, :, :] , ifs[:, i:i+nsteps, :, :], time, ml_levels) 

      plots([[ifs_metrics[m], 'royalblue', 'IFS (ml)'], 
          [atrep_target_metrics[m], 'red', 'AtmoRep (ml)']], x_label = field, y_label = metric)

      diagnostic_plots([[ifs_metrics[m], 'royalblue', 'IFS (ml)'], 
          [atrep_target_metrics[m], 'red', 'AtmoRep (ml)']], x_label = field, y_label = metric)
      write_csv(metric, field,  np.mean(atrep_target_metrics[m], axis = 0), ml_levels, steps, label = "atmorep")
      write_csv(metric, field, np.mean(ifs_metrics[m], axis = 0), ml_levels, steps, label = "ifs")

##########################################################

def create_score_cards():

  create_directory_tree(out_dir)

  metrics = ["RMSE", "ACC", "SSR"] # "L2",  "L1", "NRMSE"] #"CRPS"

  # load dataset and create DataFrame ready to create heatmap
  fields = ["velocity_u", "velocity_v", "velocity_z", "temperature",  "specific_humidity"]
  field_name = ["U component", "V component", "Z component",  "Temperature", "Specific Humidity"]

  for metric in metrics:
    print( "Info:: Computing: ", metric)
    data_df1 = [pd.read_csv("./{}/csv/{}_{}_atmorep.csv".format(out_dir, metric, field)) for field in fields]
    data_df1 = [data.loc[:, ~data.columns.str.contains('^Unnamed')].set_index("level") for data in data_df1]

    data_ifs = [pd.read_csv("./{}/csv/{}_{}_ifs.csv".format(out_dir, metric, field)) for field in fields]
    data_ifs = [data.loc[:, ~data.columns.str.contains('^Unnamed')].set_index("level") for data in data_ifs]

    #subtract 
    data_df = [data_df1[i].subtract(data_ifs[i]) for i in range(len(data_df1))]
    # create heatmap seaborn
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    new_cmap = truncate_colormap(plt.get_cmap("coolwarm"), 0.15, 0.85)
    cbar_kws = {"orientation":"vertical", 
                "shrink":1,
                'extend':'min', 
                'extendfrac':0.1, 
                "drawedges":True,
                "format": formatter,
              } # color bar keyword arguments
    
    fig, axs = plt.subplots(nrows = 3, ncols = 2, sharex=True, sharey=True, figsize=(7,7)) #,  gridspec_kw=dict(width_ratios=[4,0.2]))

    cbar_ax = fig.add_axes([.8, .15, .035, .67])
    
    for i, field in enumerate(fields):
      print("orig", data_df1[i])
      print("ifs", data_ifs[i])
      print("diff", data_df[i])
      ax = axs.flat[i]
      sns.heatmap(data_df[i], vmin = 0, vmax = 1, cmap=new_cmap, annot = False, linewidth = 2, cbar= i == 0, cbar_kws=cbar_kws, ax = ax, cbar_ax=None if i else cbar_ax)
      ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
      ax.set_ylabel("", fontsize = 20)
      ax.set_title(field_name[i])
    axs.flat[-1].axis('off')
    fig.tight_layout(rect=[0., 0, .75, 1])
    plt.savefig(out_dir+"/score_card_{}.png".format(metric))
    plt.savefig(out_dir+"/score_card_{}.pdf".format(metric))
    plt.close()

if __name__ == '__main__':
  # compute_metrics_for_score_card()
  # create_score_cards()

