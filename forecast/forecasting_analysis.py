import numpy as np
import xarray as xr
import properscoring as ps
import pandas as pd

from utils.utils import *
from utils.plotting import *
from datetime import datetime 
from forecasting_utils import *

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

fields = ["velocity_u", "velocity_v","temperature",  "specific_humidity"]
field_name = ["U component", "V component",  "Temperature", "Specific Humidity"]

metrics_names = ["ACC", "RMSE"] #"L2", "L1",NRMSE

model_ids = [

  #Overlap 2/6 days 1/10
  ("1bev7kz5", 12 ), #-> Dec 2018 
  ("1b73ccfv", 11 ), #-> Nov 2018 
  ("11tz2eld", 10 ), #-> Oct 2018 
  ("17ux1ydg",  9), #-> Sept 2018 
  ("2ppi74vj",  8), #-> Aug 2018 
  ("2z162dua",  7), #-> Jul 2018 
  ("wvu14lhp",  6), #-> Jun 2018
  ("4n3338mn",  5), #-> May 2018
  ("22q0mipv",  4), #-> Apr 2018
  ("2l2bpssq",  3), #-> Mar 2018
  ("hgm0dyl6",  2), #-> Feb 2018
  ("2390cnz0",  1), #-> Jan 2018
  # #
  # ##Overlap 2/6 days 10/20
  ("1xb1q5dw", 12),  # - Dec 2018 DONE 
  ("ni9of2y4", 11),  # - Nov 2018 DONE
  ("2bobd4w4", 10),  # - Oct 2018 DONE
  ("295gsuzb", 9 ),  # - Sep 2018 DONE
  ("2sfupztl", 8 ),  # - Aug 2018 DONE
  ("3qddvzqm", 7 ),  # - Jul 2018 DONE
  ("3qq12yic", 6 ),  # - Jun 2018 DONE
  ("208hb10q", 5 ),  # - May 2018 DONE
  ("2d8hfx2i", 4 ),  # - Apr 2018 DONE
  ("310k3ass", 3 ),  # - Mar 2018 DONE
  ("285za13m", 2 ),  # - Feb 2018 DONE
  ("10rmasb7", 1 ), # - Jan 2018 DONE
  #
  ##Overlap 2/6 days 20/30
  ("3heovh02", 12), # - Dec 2018 DONE
  ("nfsu25mo", 11), # - Nov 2018 DONE
  ("upg1khdx", 10), # - Oct 2018 DONE
  ("1ex99424", 9), # - Sep 2018 DONE
  ("1q3x43pi", 8), # - Aug 2018 DONE
  ("29lyxb22", 7), # - Jul 2018 DONE
  ("dhw054ko", 6), # - Jun 2018 DONE
  ("ssow12oo", 5), # - May 2018 DONE
  ("29z0hb21", 4), # - Apr 2018 DONE
  ("8sjhbuno", 3), # - Mar 2018 DONE
  ("zngdo2z9", 2), # - Feb 2018 DONE
  ("3rr4x9fs", 1) # - Jan 2018 DONE

]

nft_model_ids = [
  ("3tw5hkxd", 1), # -> Jan 2018 
  ("x2gql37i", 2), # -> Feb 2018
  ("b43cuolw", 3), # -> Mar 2018
  ("2uz27rfr", 4), # -> Apr 2018
  ("scnpqjh8", 5), # -> May 2018
  ("5fqvxxtl", 6), # -> Jun 2018
  ("37dbay1q", 7), # -> Jul 2018
  ("h3t5wejp", 8), # -> Aug 2018
  ("3hly57fj", 9), # -> Sep 2018
  ("1c8uuonf", 10), # -> Oct 2018
  ("3s8rvbmv", 11), # -> Nov 2018
  ("1a8a16ji", 12), # -> Dec 2018
]

year = 2018
out_dir = './FINAL_'+str(year) + '_all_year_2_6_overlap'
# out_dir = './NFT_'+str(year) + '_all_year_2_6_overlap'

cat = np.concatenate

def get_atmorep(field, model_list):
    nft_all_metrics = []
    for model_id, m in model_list:

      fname_atrep = ["./netcdf/atmorep_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format(field, model_id, year, m, ml_level) for ml_level in ml_levels]
      fname_atrep_target = ["./netcdf/atmorep_target_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format( field, model_id, year, m, ml_level) for ml_level in ml_levels]
    
      #atmorep
      atrep_target = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep_target], dim = "vlevel") 
      atrep = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep], dim = "vlevel")
      datetimes = atrep_target.time.values
      steps = np.sort(np.unique(atrep_target.step))

      nsteps = len(steps)
      ntimes = len(datetimes)
      nlevels = len(ml_levels)

      atrep_target = atrep_target.transpose('time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values
      print("Info:: AtmoRep target - Opened.")
      atrep = atrep.transpose('time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values 
      print("Info:: AtmoRep - Opened.")

       #initialize the errors
      shape = [len(metrics_names), len(datetimes), nlevels, nsteps]
      nft_metrics = np.zeros(shape)
      
      #sanity check: 2D image of the first timestamp
      #for (data, label) in [(atrep, "atmorep"), (atrep_target, "atmorep_target")]: #, (pangw, "pangw"), (ifs,"ifs"), (era5_ml, "era5_ml"), (era5_pl, "era5_pl")]:
      #diagnostic_plots_2d(out_dir, field, atrep_target, atrep, 0, 0, 0, 'atmorep') # label)

      for m, metric in enumerate(metrics_names):
        for i, time in enumerate(datetimes):
          nft_metrics[m, i] = get_metrics(metric, field, atrep_target[i], atrep[i], time, "ml") 

    
      nft_all_metrics.append(nft_metrics.swapaxes(0, 1))

    #concateate all months
    return cat(nft_all_metrics)


def forecast_metrics():

  create_directory_tree(out_dir)

  for field in fields: 
  
    pangw_all_metrics = []
    ifs_all_metrics = []
    atmorep_all_metrics = []

    for model_id, m in model_ids:

      fname_atrep = ["./netcdf/atmorep_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format(field, model_id, year, m, ml_level) for ml_level in ml_levels]
      fname_atrep_target = ["./netcdf/atmorep_target_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format( field, model_id, year, m, ml_level) for ml_level in ml_levels]
    
      #atmorep
      atrep_target = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep_target], dim = "vlevel")
      atrep = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep], dim = "vlevel")
     
      datetimes = atrep_target.time.values
      steps = np.sort(np.unique(atrep_target.step))

      nsteps = len(steps)
      ntimes = len(datetimes)
      nlevels = len(ml_levels)

      atrep_target = atrep_target.transpose('time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values
      print("Info:: AtmoRep target - Opened.")
      atrep = atrep.transpose('time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values 
      print("Info:: AtmoRep - Opened.")

      #panguweather
      fname_pangw = "/p/scratch/atmo-rep/data/Pangu-Weather/output_panguweather_6h_forecast_{}{:02d}.nc".format(year, m) 
      pangw = xr.open_dataset(fname_pangw, engine = "netcdf4").sel(isobaricInhPa=pl_levels,
                time = datetimes).transpose('time', 'isobaricInhPa', 'step', 'latitude', 'longitude')[grib_index[field]].values
      print("Info:: Panguweather - Opened.")

      #era5
      fname_era5_ml =  ["/p/scratch/atmo-rep/data/era5/ml_levels/{}/{}/reanalysis_{}_y{}_m{:02d}_ml{}.grib".format(ml_level, field, field, year, m, ml_level) for ml_level in ml_levels]
      era5_ml  = xr.concat([xr.open_dataset(fname, engine = "cfgrib").sel(time = slice(datetimes[0], datetimes[-1]+ pd.DateOffset(hours=int(steps[-1])+1))) for fname in fname_era5_ml], 
                        dim = "vlevel").transpose('vlevel', 'time', 'latitude', 'longitude')[grib_index[field]]
    
      leadtimes_idx_era5 = np.where(np.isin(era5_ml.time.values, datetimes))[0] 
      era5_time = era5_ml.time.values                           
      era5_ml = era5_ml.values
      print("Info:: ERA5 ML - Opened.")

      fname_era5_pl =  ["/p/scratch/atmo-rep/data/era5/pl_levels/{}/{}/reanalysis_{}_y{}_m{:02d}_pl{}.grib".format(pl_level, field, field, year, m, pl_level) for pl_level in pl_levels]
      era5_pl  = xr.concat([xr.open_dataset(fname, engine = "cfgrib").sel(time = slice(datetimes[0], datetimes[-1]+ pd.DateOffset(hours=int(steps[-1]+1)))) for fname in fname_era5_pl], 
                        dim = "vlevel").transpose('vlevel', 'time','latitude', 'longitude')[grib_index[field]].values                             
                        #.values             
      print("Info:: ERA5 PL - Opened.")
      
      #IFS
      fname_ifs   = ["/p/scratch/atmo-rep/data/ifs/ml_levels/{}/{}/ifs_forecast_{}_y{}_m{:02d}_ml{}.nc".format(ml_level, field, field, year, m, ml_level) for ml_level in ml_levels]
      ifs = xr.concat([xr.open_dataset(fname, engine = "netcdf4").sel(time = datetimes) for fname in fname_ifs], 
                        dim = "vlevel").transpose('time','vlevel', 'step', 'latitude', 'longitude')[grib_index[field]]                        
      ifs_time = ifs.time.values
      ifs = ifs.values
      print("Info:: IFS - Opened.")

      #############################
      print("Info:: atmorep       ", atrep.shape)
      print("Info:: panguweather  ", pangw.shape)
      print("Info:: ERA5 ml       ", era5_ml.shape)
      print("Info:: ERA5 pl       ", era5_pl.shape)
      print("Info:: IFS           ", ifs.shape)    
      print("Info:: leadtimes ERA5", leadtimes_idx_era5.shape)    
      print("Info:: nsteps        ", nsteps)
      
      #initialize the errors
      shape = [len(metrics_names), len(datetimes), nlevels, nsteps]
      atrep_target_metrics = np.zeros(shape)
      pangw_metrics = np.zeros(shape)
      ifs_metrics = np.zeros(shape)

      #sanity check: 2D image of the first timestamp
      #for (data, label) in [(atrep, "atmorep"), (atrep_target, "atmorep_target")]: #, (pangw, "pangw"), (ifs,"ifs"), (era5_ml, "era5_ml"), (era5_pl, "era5_pl")]:
      #diagnostic_plots_2d(out_dir, field, atrep_target, atrep, 0, 0, 0, 'atmorep') # label)

      for m, metric in enumerate(metrics_names):
        for i, time in enumerate(datetimes):
        
          era5_fcst_lt = leadtimes_idx_era5[i]+1
          print("Running step:: ", time, era5_time[era5_fcst_lt], ifs_time[i] )
          atrep_target_metrics[m, i] = get_metrics(metric, field, atrep_target[i], atrep[i], time, "ml") 
          pangw_metrics[m, i]        = get_metrics(metric, field, era5_pl[:, era5_fcst_lt:era5_fcst_lt+nsteps, :, :] , pangw[i, :, :nsteps, :, :], time, "pl") 
          ifs_metrics[m, i]          = get_metrics(metric, field, era5_ml[:, era5_fcst_lt:era5_fcst_lt+nsteps, :, :] , ifs[i, :, :nsteps, :, :], time, "ml") 
      
      pangw_all_metrics.append(pangw_metrics.swapaxes(0, 1))
      ifs_all_metrics.append(ifs_metrics.swapaxes(0, 1))
      atmorep_all_metrics.append(atrep_target_metrics.swapaxes(0, 1))

    #concateate all months
    pangw_all_metrics = cat(pangw_all_metrics)
    ifs_all_metrics = cat(ifs_all_metrics)
    atmorep_all_metrics = cat(atmorep_all_metrics)

    pangw_all_metrics.tofile(out_dir+'/pangw_all_metrics_'+field+'.dat')
    ifs_all_metrics.tofile(out_dir+'/ifs_all_metrics_'+field+'.dat')
    atmorep_all_metrics.tofile(out_dir+'/atmorep_all_metrics_'+field+'.dat')
    
    print(pangw_all_metrics.shape)
    print(ifs_all_metrics.shape)
    print(atmorep_all_metrics.shape)

    nft = get_atmorep(field, nft_model_ids)[..., :3] 
    nft.tofile(out_dir+'/nft_all_metrics.dat')

    for m, metric in enumerate(metrics_names):
      plots(out_dir, [[ifs_all_metrics[:, m], 'royalblue', 'IFS (ml)'], 
          [pangw_all_metrics[:, m], 'black', 'PanguWeather (pl)'], 
          [atmorep_all_metrics[:, m], 'red', 'AtmoRep (ml)']], x_label = field, y_label = metric) 

      single_level_plots(out_dir, [[ifs_all_metrics[:, m], 'royalblue', 'IFS (ml)'], 
          [pangw_all_metrics[:, m], 'black', 'PanguWeather (pl)'], 
          [atmorep_all_metrics[:, m], 'red', 'AtmoRep - 6h (ml)'], 
          [nft[:, m], 'darkseagreen', 'AtmoRep non-fine tuned - 3h (ml)']], x_label = field, y_label = metric) 
     
   
###########################################################################

def quick_plotting():
    in_dir = './NFT_'+str(year) + '_all_year_2_6_overlap'
    for field in fields:
      pangw_all_metrics = np.fromfile(in_dir+'/pangw_all_metrics_'+field+'.dat').reshape(-1, 2, 5, 6)
      ifs_all_metrics = np.fromfile(in_dir+'/ifs_all_metrics_'+field+'.dat').reshape(-1, 2, 5, 6)
      atmorep_all_metrics = np.fromfile(in_dir+'/atmorep_all_metrics_'+field+'.dat').reshape(-1, 2, 5, 6)
      
      print(pangw_all_metrics.shape)
      print(ifs_all_metrics.shape)
      print(atmorep_all_metrics.shape)

      nft = get_atmorep(field, nft_model_ids)[..., :3] #None
      nft.tofile(out_dir+'/nft_all_metrics_'+field+'.dat')
      nft_all_metrics = np.fromfile(in_dir+'/nft_all_metrics_'+field+'.dat').reshape(-1, 2, 5, 3)
      print(nft_all_metrics.shape)

      for m, metric in enumerate(metrics_names):
        plots(out_dir, [[ifs_all_metrics[:, m], 'royalblue', 'IFS (ml)'], 
            [pangw_all_metrics[:, m], 'black', 'PanguWeather (pl)'], 
            [nft_all_metrics[:, m], 'darkseagreen', 'AtmoRep non-fine tuned - 3h (ml)'], 
            [atmorep_all_metrics[:, m], 'red', 'AtmoRep (ml)']], x_label = field, y_label = metric) 

        single_level_plots(out_dir, [[ifs_all_metrics[:, m], 'royalblue', 'IFS (ml)'], 
            [pangw_all_metrics[:, m], 'black', 'PanguWeather (pl)'], 
            [atmorep_all_metrics[:, m], 'red', 'AtmoRep - 6h (ml)'], 
            [nft_all_metrics[:, m], 'darkseagreen', 'AtmoRep non-fine tuned - 3h (ml)']
            ], x_label = field, y_label = metric) 


def ensemble_analysis():
    
    create_directory_tree(out_dir)

    hours = [ 0, 3, 6, 9, 12, 15, 18, 21] 

    for field in fields: 

      atrep_all_crps = []
      atrep_all_ssr = []
      era5_all_crps = []
      era5_all_ssr = []

      for model_id, m in model_ids:
        #atmorep target
        fname_atrep_target = ["./netcdf/atmorep_target_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format( field, model_id, year, m, ml_level) for ml_level in ml_levels]
        atrep_target = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep_target], dim = "vlevel") 
        datetimes = atrep_target.time.values
        print(datetimes)
        
        steps = np.sort(np.unique(atrep_target.step))
        nsteps = len(steps)
        atrep_target = atrep_target.transpose('time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values

        #atmorep ensemble
        fname_atrep_ensemble = ["./netcdf/atmorep_ensemble_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format( field, model_id, year, m, ml_level) for ml_level in ml_levels]
        atrep_ens = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep_ensemble], dim = "vlevel") 
        atrep_ens = atrep_ens.transpose('member', 'time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values

        #era5
        #IMPORTANT:: NEED TO FILTER BY HOUR AND KEEP ONLY 3h-spaced DATA BECAUSE "em" AND "es" ARE 3h-spaced.  
        fname_era5_ml =  ["/p/scratch/atmo-rep/data/era5/ml_levels/{}/{}/reanalysis_{}_y{}_m{:02d}_ml{}.grib".format(ml_level, field, field, year, m, ml_level) for ml_level in ml_levels]
        era5_ml  = xr.concat([xr.open_dataset(fname, engine = "cfgrib").sel(time = slice(datetimes[0], datetimes[-1]+ pd.DateOffset(hours=int(steps[-1]+1)))) for fname in fname_era5_ml], 
                          dim = "vlevel")
        era5_ml = era5_ml.sel(time = era5_ml.time.dt.hour.isin(hours)) #filter by hour
        datetimes_era5 = [pd.to_datetime(x) for x in np.unique(era5_ml.time)]
       
        leadtimes_idx_era5 = np.where(np.isin(era5_ml.time.values, datetimes))[0] 
       
        era5_ml = era5_ml.transpose('vlevel', 'time', 'latitude', 'longitude')[grib_index[field]].values                             
        print("Info:: ERA5 ML - Opened.")

        #ERA5 em
        steps_era5 = [0]+ [s for s in steps if (s) % 3 == 0] 
        print(steps_era5)
        nsteps_era5 = len(steps_era5)
        fname_era5_es   = ["/p/scratch/atmo-rep/data/era5_ensemble/ml_levels/{}/{}/reanalysis_{}_y{}_m{:02d}_ml{}_es.grib".format(ml_level, field, field, year, m, ml_level) for ml_level in ml_levels]
        era5_es = xr.concat([xr.open_dataset(fname, engine = "cfgrib").sel(time = datetimes_era5) for fname in fname_era5_es], 
                          dim = "vlevel")
        era5_es = era5_es.sel(time = era5_es.time.dt.hour.isin(hours)) #select every 3h
        datetimes_era5_es = [pd.to_datetime(x) for x in np.unique(era5_es.time)]
        # print(datetimes_era5_es)
        era5_es = era5_es.transpose('vlevel', 'time','latitude', 'longitude')[grib_index[field]].values    
        print("Info:: ERA5 em ML - Opened.")

        #ERA5 es
        fname_era5_em   = ["/p/scratch/atmo-rep/data/era5_ensemble/ml_levels/{}/{}/reanalysis_{}_y{}_m{:02d}_ml{}_em.grib".format(ml_level, field, field, year, m, ml_level) for ml_level in ml_levels]
        era5_em = xr.concat([xr.open_dataset(fname, engine = "cfgrib").sel(time = datetimes_era5) for fname in fname_era5_em], 
                          dim = "vlevel")
        era5_em = era5_em.sel(time = era5_em.time.dt.hour.isin(hours))
        # print(np.unique(era5_em.time))     
        datetimes_era5_em = [pd.to_datetime(x) for x in np.unique(era5_em.time)]   
        # print(datetimes_era5_em)
        era5_em = era5_em.transpose('vlevel', 'time','latitude', 'longitude')[grib_index[field]].values    
        print("Info:: ERA5 es ML - Opened.")
        
        mean_atrep = np.mean(atrep_ens, axis = 0)
        std_atrep = np.std(atrep_ens, axis = 0)

        #matrics: CRPS
        atrep_crps = np.zeros([len(datetimes), len(ml_levels), nsteps])
        era5_crps = np.zeros([len(leadtimes_idx_era5), len(ml_levels), nsteps_era5])
        
        #metrics: SSR
        atrep_ssr = np.zeros([len(datetimes), len(ml_levels), nsteps])
        era5_ssr = np.zeros([len(leadtimes_idx_era5), len(ml_levels), nsteps_era5])

        #Compute metrics:
        for i, time in enumerate(datetimes):
          print("Running step:: ", time)
          atrep_crps[i] = calc_crps(atrep_target[i], mean_atrep[i], std_atrep[i])
          atrep_ssr[i] = calc_ssr(atrep_target[i], mean_atrep[i], std_atrep[i])
        print("Info:: AtmoRep CRPS+SSR Ended.")

        #for i, time in enumerate(datetimes_era5):
        for i, era5_fcst_lt in enumerate(leadtimes_idx_era5):
          #era5_fcst_lt = leadtimes_idx_era5[i]
          print("Running step:: ", datetimes_era5[era5_fcst_lt], datetimes_era5_em[era5_fcst_lt:era5_fcst_lt+nsteps_era5])
          era5_crps[i] = calc_crps(era5_ml[:, era5_fcst_lt:era5_fcst_lt+nsteps_era5, :, :], era5_em[:, era5_fcst_lt:era5_fcst_lt+nsteps_era5, :, :], era5_es[:, era5_fcst_lt:era5_fcst_lt+nsteps_era5, :, :])
          era5_ssr[i]  = calc_ssr(era5_ml[:, era5_fcst_lt:era5_fcst_lt+nsteps_era5, :, :], era5_em[:, era5_fcst_lt:era5_fcst_lt+nsteps_era5, :, :], era5_es[:, era5_fcst_lt:era5_fcst_lt+nsteps_era5, :, :])
        print("Info:: ERA5 CRPS+SSR Ended.")

        atrep_all_crps.append(atrep_crps)
        atrep_all_ssr.append(atrep_ssr)
        era5_all_crps.append(era5_crps)
        era5_all_ssr.append(era5_ssr)


      atrep_all_crps = cat(atrep_all_crps)
      atrep_all_ssr = cat(atrep_all_ssr)
      era5_all_crps = cat(era5_all_crps)
      era5_all_ssr = cat(era5_all_ssr)

      #plot
      plots_ensemble(out_dir, field, atrep_all_crps, era5_all_crps, atrep_all_ssr, era5_all_ssr)
     
###########################################################################

def forecast_analysis_non_fine_tuned():

  month = 1
  year = 2018

  epochs = [
  ("1jzqblji", "training day : 5", "red"), 
  ("2krzsewn",  "training day : 4", "magenta"),  
  ("31pkwocr", "training day : 3", "orange"), 
  ("3tdvarjs", "training day : 2", "green"),  
  ("1nisksuz", "training day : 1", "blue"), 
  ("3tw5hkxd", "non-fine-tuned", "black")]

  #Non fine tuned model
  create_directory_tree(out_dir)
  metrics_names = ["ACC", "RMSE"] #"L2", "L1",NRMSE

  for field in fields: 
    atrep_all_metrics = {"ACC" : [], "RMSE" : []}
    for e, (model, label, color) in enumerate(epochs): 
      print("Info:: Computing metrics for model ", model, label)
    ###############################

      fname_atrep = ["./netcdf/atmorep_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format(field, model , year, month, ml_level) for ml_level in ml_levels]
      fname_atrep_target = ["./netcdf/atmorep_target_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format( field, model, year, month, ml_level) for ml_level in ml_levels]
    
      #AtmoRep:: fine tuned
      atrep_target = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep_target], dim = "vlevel") #.isel(step = np.arange(nsteps))
      atrep = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep], dim = "vlevel") #.isel(time = range(len(datetimes1))) #.isel(step = np.arange(nsteps))
      datetimes = [pd.to_datetime(x) for x in np.unique(atrep_target.time.values)]
      steps = np.sort(np.unique(atrep_target.step))
      nsteps = len(steps)
      nlevels = len(ml_levels)

      atrep_target = atrep_target.transpose('time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values
      print("Info:: AtmoRep target - Opened.")
      atrep = atrep.transpose('time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values 
      print("Info:: AtmoRep - Opened.")

      #initialize the errors
      atrep_metrics = np.zeros([len(metrics_names), len(datetimes),  nlevels,nsteps])

      #sanity check: 2D image of the first timestamp
      # for (data, label) in [(atrep_target, "atmorep"), (pangw, "pangw"), (ifs,"ifs"), (era5_ml, "era5_ml"), (era5_pl, "era5_pl")]:
      #  diagnostic_plots_2d(field, data, 0, 0, 0, label)

      for m, metric in enumerate(metrics_names):
        for i, time in enumerate(datetimes):
          #print("Running step:: ", time)
          atrep_metrics[m, i] = get_metrics(metric, field, atrep_target[i], atrep[i], time, "ml") 

        atrep_all_metrics[metric].append([atrep_metrics[m], color, label])

    for m, metric in enumerate(metrics_names):
      plots(out_dir, atrep_all_metrics[metric], x_label = field, y_label = metric)
      single_level_plots(out_dir, atrep_all_metrics[metric], x_label = field, y_label = metric)
    
def forecast_2d_plots():

  create_directory_tree(out_dir)
  model_id = "3qq12yic"
  m = 6  # - Jun 2018 DONE

  cmaps= {
    "specific_humidity" : ["PuBuGn",1], 
    "temperature" : ["PuBuGn",1],  #"bwr", 
    "velocity_u" : ["PuBuGn",1],  #"twilight_shifted", 
    "velocity_v" : ["PuBuGn",1] 
  }

  for field in fields: 

    fname_atrep = ["./netcdf/atmorep_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format(field, model_id, year, m, ml_level) for ml_level in ml_levels]
    fname_atrep_target = ["./netcdf/atmorep_target_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc".format( field, model_id, year, m, ml_level) for ml_level in ml_levels]
  
    #atmorep
    atrep_target = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep_target], dim = "vlevel").sel(time = ['2018-06-15 12:00:00', '2018-06-15 00:00:00']) 
    atrep = xr.concat([xr.open_dataset(fname, engine = "netcdf4") for fname in fname_atrep], dim = "vlevel").sel(time = ['2018-06-15 12:00:00', '2018-06-15 00:00:00']) 
    datetimes = atrep_target.time.values
    steps = np.sort(np.unique(atrep_target.step))

    nsteps = len(steps)
    ntimes = len(datetimes)
    nlevels = len(ml_levels)

    atrep_target = atrep_target.transpose('time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values
    print("Info:: AtmoRep target - Opened.")
    atrep = atrep.transpose('time', 'vlevel', 'step', 'lat', 'lon')[grib_index[field]].values 
    print("Info:: AtmoRep - Opened.")

    diagnostic_plots_2d(out_dir, field, atrep_target, atrep, 0, 0, 0, 'atmorep', cmap=cmaps[field][0], alpha = cmaps[field][1])



##########################################################################

if __name__ == '__main__':
  # forecast_analysis_non_fine_tuned() #plot as a function of the training day
  # forecast_metrics()   # plot ACC/RMSE
  # ensemble_analysis()  # plot CRPS/SSR
  # forecast_2d_plots()  # plot 2D predictions
  quick_plotting()       # adjust plot style