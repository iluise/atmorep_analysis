# SPDX-FileCopyrightText: 2023 
#
# SPDX-License-Identifier: MIT

"""
Methods that are shared among the forecasting analysis scripts.
"""

__author__ = "Ilaria Luise"
__email__ = "ilaria.luise@cern.ch"
__date__ = "2023-08-27"
__update__ = "2023-08-27"

import numpy as np
import xarray as xr
import properscoring as ps
import pandas as pd

from utils.utils import *
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
from matplotlib.ticker import FormatStrFormatter

field_name = {"velocity_u": "U component",
              "velocity_v": "V component",  
              "temperature": "Temperature", 
              "specific_humidity": "Specific Humidity"}

ml_levels = [137, 123, 114, 105, 96]
pl_levels = [1000, 925, 850, 700, 500]

def create_directory_tree(dir_out):
    if not os.path.exists(dir_out):
     os.makedirs(dir_out)
    if not os.path.exists(dir_out+"/plots"):
     print("Info:: plots stored in: ", dir_out+"/plots")
     os.makedirs( dir_out+"/plots")
    if not os.path.exists(dir_out+"/csv"):
     print("Info:: csv stored in: ", dir_out+"/csv")
     os.makedirs( dir_out+"/csv")

########################################

def get_units_mse(field):
  units = {"temperature": "K^2", 
           "velocity_u": "m$^2$ s$^{-2}$", 
           "velocity_v": "m$^2$ s$^{-2}$", 
           "velocity_z": "Pa$^2$ s${-2}$", 
           "specific_humidity": "g$^2$ kg$^{-2}$", 
          }

  return units[field]

########################################

def get_metrics(type, field, targets, preds, datetime, lvl_type, apply_weights = True) :

  vlevels = None
  if lvl_type == "pl":
    vlevels = pl_levels
  elif lvl_type == "ml":
    vlevels = ml_levels
  else:
    assert False, f'Error:: Unknown level type. Please use ml or pl.'

  if(np.isnan(preds).any()):
    print("Info:: NaNs detected and masked. Ratio: {:.2E}".format(preds[np.where(np.isnan(preds))].flatten().shape[0]/preds.flatten().shape[0])) 
    preds[np.where(np.isnan(preds))] = targets[np.where(np.isnan(preds))]
   
  #metrics: 0: RMSE, 1: errs_l2, 2: errs_l1, 3: ACC.
  metrics = np.zeros( [targets.shape[0], targets.shape[1]] )
  if type == "NRMSE":
    metrics = calc_norm_rmse(targets, preds)
  elif type == "RMSE":
    metrics = calc_rmse(targets, preds, apply_weights)
  elif type == "L1":
    metrics = calc_L1(targets, preds)
  elif type == "L2":
    metrics = calc_L2(targets, preds)
  elif type == "ACC":
    metrics = calc_acc(field, targets, preds, datetime, vlevels, apply_weights)
  else:
    print("Error:: unknown metric. Supported: [RMSE, L1, L2, ACC]. ")
  
  return metrics

########################################

def get_weights(weight = True):
  if(weight):
    lat_sum = np.sum(np.cos(np.arange( 90.* np.pi/180., -90. * np.pi/180., -np.pi/721.)))
   
    theta_weight = np.array([721 * np.cos(w)/lat_sum for w in np.arange( 90. * np.pi/180. , -90. * np.pi/180., -np.pi/721.)], dtype = np.float32)
    theta_weight = np.repeat(theta_weight[:, np.newaxis], 1440, axis = 1)
  else:
    theta_weight = 1

  return theta_weight

########################################

def calc_rmse(targets, preds, weight = True):
  metrics = np.zeros( [targets.shape[0], targets.shape[1]] )
  theta_weight = get_weights(weight)

  for ilevel in range( targets.shape[0]) :
    for i in range(targets.shape[1]) :
      metrics[ilevel,i] = np.sqrt(np.sum( theta_weight * ((preds[ilevel,i] - targets[ilevel,i] )**2)) / (targets.shape[2] * targets.shape[3]))
  return metrics

########################################

def calc_L1(targets, preds):
  metrics = np.zeros( [targets.shape[0], targets.shape[1]] )
  for ilevel in range( targets.shape[0]) :
    for i in range(targets.shape[1]) :
      metrics[ilevel,i] = np.linalg.norm( (targets[ilevel,i] - preds[ilevel,i]).flatten(), 1)
  metrics /= (targets.shape[2] * targets.shape[3])
  return metrics

########################################

def calc_L2(targets, preds):
  metrics = np.zeros( [targets.shape[0], targets.shape[1]] )
  for ilevel in range( targets.shape[0]) :
    for i in range(targets.shape[1]) :
      metrics[ilevel,i] = np.linalg.norm( (targets[ilevel,i] - preds[ilevel,i]).flatten(), 2)
  metrics /= (targets.shape[2] * targets.shape[3])
  return metrics

########################################

def calc_acc(field, targets, preds, dt, vlevels, weight = True):
  """
  Calculate acc ealuation metric of forecast data w.r.t reference data
  :param targets: reference data (xarray with dimensions [batch, fore_hours, lat, lon])
  :param preds: forecasted data (xarray with dimensions [batch, fore_hours, lat, lon])
  :param datetimes: list of datetimes64
  :return: averaged acc for each example same shape as lvl x steps.
  """ 
  month = pd.Timestamp(dt).month
  theta_weight = get_weights(weight)
  acc = np.ones([targets.shape[0], targets.shape[1]])*np.nan
  for lvl_idx, lvl in enumerate(vlevels):
    for dt_idx in range(targets.shape[1]):
      mean = get_climatological_mean(field, month, lvl)
      if (mean.mean() == 0) and (mean.std() == 0):
        mean = compute_climatological_mean(field, month, lvl)
      mean = mean[:targets.shape[-2], :targets.shape[-1]]
      
      img1_ = targets[lvl_idx, dt_idx] - mean
      img2_ = preds[lvl_idx, dt_idx] - mean

      cor1 = np.sum(theta_weight*img1_*img2_)

      cor2 = np.sqrt(np.sum(theta_weight * (img1_**2))*np.sum(theta_weight *(img2_**2)))
      if(np.sum(theta_weight * (img1_**2))*np.sum(theta_weight *(img2_**2)) < 0):
        print("ACC denominator is NaN. Error.")
        print(np.sum(theta_weight * (img1_**2)), np.sum(theta_weight *(img2_**2))) 
      acc[lvl_idx, dt_idx] = cor1/cor2
  return acc

########################################

def calc_crps(targets, ens_mean, ens_std):
  # see https://pypi.org/project/properscoring/
  crps = ps.crps_gaussian(targets, mu=ens_mean, sig=ens_std)
  crps = np.array([crps[lvl_idx, dt_idx].mean() for lvl_idx,dt_idx in itertools.product(range(targets.shape[0]), range(targets.shape[1]))])
  return crps.reshape([targets.shape[0], targets.shape[1]])

########################################

def calc_ssr(targets, ens_mean, ens_std, weight = True):
  theta_weight = get_weights(weight)
  spread = np.zeros( [targets.shape[0], targets.shape[1]] )
  for ilevel in range( targets.shape[0]) :
    for i in range(targets.shape[1]) :
      spread[ilevel, i] = np.sqrt(np.sum( theta_weight * ((ens_std[ilevel,i])**2)) / (targets.shape[2] * targets.shape[3]))
  return spread/calc_rmse(targets, ens_mean, weight = weight)

#############################################################

def plot_on_map(data, field, ax, cmap = "RdBu", norm = None, zrange = [0.,0.1]):
    """
    plot on a world map
    """
    ax.set_global()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    #see: https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
    pos1 = ax.get_position() # get the original position 
    pos2 = [pos1.x0 - 0.04, pos1.y0,  pos1.width, pos1.height] 
    ax.set_position(pos2) # set a new position

    if(norm != None):
      im = plt.imshow(data, cmap=cmap, extent=[-180,180,-90,90], norm=norm) #or use plt.pcolor
    else:
      im = plt.imshow(data, cmap=cmap, extent=[-180,180,-90,90], vmin=zrange[0],  vmax=zrange[1]) #or use plt.pcolor
    return im, ax

#############################################################

def diagnostic_plots_2d(out_dir, field,  targets,  preds, vlevel, timestep, step, label, cmap = "PuBuGn", alpha = 1):
  fig = plt.figure(figsize=(14, 6))
  ax = []
  text_kwargs = dict(ha='center', va='center', fontsize = 15 )
  plt.text(0.5, 0.83, field_full_name[field],  **text_kwargs)
  plt.axis("off")
  if(len(targets.shape) == 5 ):
    mean_target = targets[vlevel, timestep, step].mean()
    std_target = targets[vlevel, timestep, step].std()
   
    mean_preds = preds[vlevel, timestep, step].mean()
    std_preds = preds[vlevel, timestep, step].std()

    save_ims( out_dir, field, preds[vlevel, timestep, step], 'preds', min_val = (mean_preds - std_preds*3), max_val = (mean_preds + std_preds*3))
    save_ims( out_dir, field, targets[vlevel, timestep, step], 'targets', min_val = (mean_preds - std_preds*3), max_val = (mean_preds + std_preds*3))
    save_ims( out_dir, field, targets[vlevel, timestep, step]-preds[vlevel, timestep, step], 'diff')
   
    ax.append(fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree(central_longitude=180)))
    im, ax[0] = plot_on_map(preds[vlevel, timestep, step], field, ax = ax[0], cmap = cmap,zrange = [mean_preds - std_preds*2, mean_preds + std_preds*2])

    ax.append(fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree(central_longitude=180)))
    im, ax[1] = plot_on_map(targets[vlevel, timestep, step], field,ax = ax[1],  cmap = cmap,zrange = [mean_preds - std_preds*2, mean_preds + std_preds*2])
  
  else:
    mean_target = targets[vlevel, timestep].mean()
    std_target = targets[vlevel, timestep].std()
   
    mean_preds = preds[vlevel, timestep].mean()
    std_preds = preds[vlevel, timestep].std()
  
    save_ims( out_dir, field, preds[vlevel, timestep, step], 'preds', min_val = (mean_preds - std_preds*2), max_val = (mean_preds + std_preds*2))
    save_ims( out_dir, field, targets[vlevel, timestep, step], 'targets', min_val = (mean_preds - std_preds*2), max_val = (mean_preds + std_preds*2))
    save_ims( out_dir, field, targets[vlevel, timestep, step]-preds[vlevel, timestep, step], 'diff')

    ax.append(fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree(central_longitude=180)))
    im, ax[0] = plot_on_map(preds[vlevel, timestep], field, ax = ax[0], cmap = cmap,zrange = [mean_preds - std_preds*3, mean_preds + std_preds*2])
    
    ax.append(fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree(central_longitude=180)))
    im, ax[1] = plot_on_map(targets[vlevel, timestep], field, ax = ax[1], cmap = cmap,zrange = [mean_preds - std_preds*3, mean_preds + std_preds*2])
    
  cbar = fig.colorbar(im, shrink=0.7, cax=fig.add_axes([0.92, 0.32, 0.02, 0.35], alpha = alpha)) #, format='%.0e')
 
  plt.title(get_units(field))
  plt.subplots_adjust(wspace=0.15)
  plt.savefig(out_dir + "/plots/{}_{}_step{}.png".format(label, field, timestep))
  plt.savefig(out_dir + "/plots/{}_{}_step{}.pdf".format(label, field, timestep))
  plt.close("all")

  return 0

########################################

def plots(out_dir, data_list, x_label, y_label, t_scale = 1): #, nft = None):
  lines = []
  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  print("Info:: plotting "+ x_label + " " + y_label)
  for i in range(data_list[0][0].shape[1]): #vlvl
    for idx, data in enumerate(data_list): #graphs
      data_mean = np.mean(data[0], axis = 0)
      data_std = np.std(data[0], axis = 0)

      plt.fill_between( (np.arange(data_mean.shape[-1])+1), data_mean[i] - data_std[i], data_mean[i] + data_std[i], color=data[1], alpha=0.1, label='_') #*(1-i/data_mean.shape[0])
      lines.append(plt.plot( (np.arange(data_mean.shape[-1])+1)*t_scale  , data_mean[i], marker='.', linestyle='-', color=data[1], alpha = 1 , linewidth=0.5 , label = data[2]))
     
  plt.legend([lines[i][0] for i in range(len(data_list)) ], [lines[i][0].get_label() for i in range(len(data_list)) ], frameon = False)
  plt.title(x_label + " for all levels")
  plt.xlabel('forecast time [h]')
  plt.ylabel(y_label)
  if(y_label == "RMSE"):
    plt.ylabel(y_label + " [" + get_units(x_label) + "]")
  if(y_label == "ACC"):
    ax.set_ylim([ax.get_ylim()[0], 1.01])
  if(y_label == "ACC" and x_label == 'specific_humidity'):
    ax.set_ylim([ax.get_ylim()[0], 1.05])
  plt.savefig(out_dir + '/prediction_'+y_label+'_' + x_label + '_all.png')
  plt.savefig(out_dir + '/prediction_'+y_label+'_' + x_label + '_all.pdf')
  plt.tight_layout()
  plt.close()

########################################

def single_level_plots(out_dir,  data_list, x_label, y_label, t_scale = 1): #, nft = None):
  lines = []

  print("Info:: plotting "+ x_label + " " + y_label)
  for i in range(data_list[0][0].shape[1]): #vlvl
    fig = plt.figure() 
    ax  = fig.add_subplot(111)
    for idx, data in enumerate(data_list): #graphs
      data_mean = np.mean(data[0], axis = 0)
      data_std = np.std(data[0], axis = 0)
      plt.fill_between( (np.arange(data_mean.shape[-1])+1), data_mean[i] - data_std[i], data_mean[i] + data_std[i], color=data[1], alpha=0.1, label='_') 
      lines.append(plt.plot( (np.arange(data_mean.shape[-1])+1)*t_scale  , data_mean[i], marker='.', linestyle='-', color=data[1], alpha = 1 , linewidth=0.5 , label = data[2]))

    ax.legend([lines[i][0] for i in range(len(data_list)) ], [lines[i][0].get_label() for i in range(len(data_list)) ], frameon = False)
    ax.set_title(field_name[x_label] + ' for model level '+ str(ml_levels[i]))
    ax.set_xlabel('forecast time [h]')
    ax.set_ylabel(y_label)
    if(y_label == "RMSE"):
      ax.set_ylabel(y_label + " [" + get_units(x_label) + "]")
    if(y_label == "ACC"):
      ax.set_ylim([ax.get_ylim()[0], 1.01])
    if(y_label == "ACC" and x_label == 'specific_humidity'):
      ax.set_ylim([ax.get_ylim()[0], 1.05])
    ax.yaxis.set_major_formatter(tkr.ScalarFormatter(useMathText=True))
    formatter = ax.yaxis.get_major_formatter() 
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    plt.tight_layout()
    plt.savefig(out_dir + '/plots/prediction_'+y_label+'_' + x_label + '_lvl'+str(ml_levels[i])+'.png')
    plt.savefig(out_dir + '/plots/prediction_'+y_label+'_' + x_label + '_lvl'+str(ml_levels[i])+'.pdf')
    plt.close()

########################################

def plots_ensemble(out_dir, field, atrep_crps, era5_crps, atrep_ssr, era5_ssr):

      atrep_crps_mean = np.mean(atrep_crps, axis = 0)
      era5_crps_mean = np.mean(era5_crps, axis = 0)

      atrep_crps_std = np.std(atrep_crps, axis = 0)
      era5_crps_std = np.std(era5_crps, axis = 0)

      atrep_ssr_mean = np.mean(atrep_ssr, axis = 0)
      era5_ssr_mean = np.mean(era5_ssr, axis = 0)

      atrep_ssr_std = np.std(atrep_ssr, axis = 0)
      era5_ssr_std = np.std(era5_ssr, axis = 0)

      print("Info:: plotting "+ field)
      lines = []
      for i in range(atrep_crps_mean.shape[0]): #vlvl
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
        ax1.fill_between( (np.arange(atrep_crps_mean.shape[-1])+1), atrep_crps_mean[i] - atrep_crps_std[i], atrep_crps_mean[i] + atrep_crps_std[i], color='darkorange', alpha=0.1, label='_')
        lines.append(ax1.plot( (np.arange(atrep_crps_mean.shape[-1])+1), atrep_crps_mean[i],   marker='.', linestyle='-', color='darkorange', linewidth=0.5 , label = "AtmoRep") ) #alpha = 1-(i/atrep_crps_mean.shape[0]) 
        lines.append(ax1.plot( (np.arange(era5_crps_mean.shape[-1]))*3, era5_crps_mean[i], marker='.', linestyle='--', color='darkorange' , linewidth=0.5 , label = 'ERA5'))
        
        ax1.legend(["_", "AtmoRep", "ERA5"], frameon = False)
        ax1.set_title(field_name[field] + ' for model level '+ str(ml_levels[i]))
        ax1.set_ylabel('CRPS')

        if(field == 'specific_humidity' and ml_levels[i] == 96):
          ax1.set_ylim([0, 1.2e-4])  
        else:
          ax1.set_ylim([0, max(np.amax(atrep_crps_mean),np.amax(era5_crps_mean))*1.1])
        ax1.yaxis.set_major_formatter(tkr.ScalarFormatter(useMathText=True))
        formatter = ax1.yaxis.get_major_formatter() #tkr.ScalarFormatter(useMathText = True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 1))

        ax2.axhline(y=1, color='black', linestyle='--', linewidth = 0.3)
        ax2.fill_between( (np.arange(atrep_ssr_mean.shape[-1])+1), atrep_ssr_mean[i] - atrep_ssr_std[i], atrep_ssr_mean[i] + atrep_ssr_std[i], color='royalblue', alpha=0.1, label='_')
        lines.append(ax2.plot( (np.arange(atrep_ssr_mean.shape[-1])+1), atrep_ssr_mean[i],   marker='.', linestyle='-', color='royalblue', linewidth=0.5 , label = "AtmoRep") ) #alpha = 1-(i/atrep_crps_mean.shape[0]) 
        lines.append(ax2.plot( (np.arange(era5_ssr_mean.shape[-1]))*3  , era5_ssr_mean[i], marker='.', linestyle='--', color='royalblue' , linewidth=0.5 , label = 'ERA5'))
        ax2.legend(["_", "_",  "AtmoRep", "ERA5"], frameon = False)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.set_xlabel('forecast time [h]')
        ax2.set_ylabel("SSR")
        ax2.set_xlim([0.97, 6.03])
        ax2.set_ylim([min(np.amin(atrep_ssr_mean),np.amin(era5_ssr_mean))*0.9, max(np.amax(atrep_ssr_mean),np.amax(era5_ssr_mean))*1.1])
        plt.savefig(out_dir + '/plots/prediction_CRPS_SSR_' + field + '_lvl'+str(ml_levels[i])+'.png')
        plt.savefig(out_dir + '/plots/prediction_CRPS_SSR_' + field + '_lvl'+str(ml_levels[i])+'.pdf')
        plt.close()

      fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
      ax1.yaxis.major.formatter.set_powerlimits((0,0))
     
      for i in range(atrep_crps_mean.shape[0]): 
        ax1.fill_between( (np.arange(atrep_crps_mean.shape[-1])+1), atrep_crps_mean[i] - atrep_crps_std[i], atrep_crps_mean[i] + atrep_crps_std[i], color='red', alpha=0.1, label='_')
        lines.append(ax1.plot( (np.arange(atrep_crps_mean.shape[-1])+1), atrep_crps_mean[i],   marker='.', linestyle='-', color='red',  alpha = 1 , linewidth=0.5 , label = "AtmoRep") )
        lines.append(ax1.errorbar( (np.arange(era5_crps_mean.shape[-1]))*3  , era5_crps_mean[i], yerr =  era5_crps_std[i], marker='P', linestyle='None', color='royalblue',  alpha = 1 , linewidth=0.5 , label = 'ERA5'))

        ax2.fill_between( (np.arange(atrep_ssr_mean.shape[-1])+1), atrep_ssr_mean[i] - atrep_ssr_std[i], atrep_ssr_mean[i] + atrep_ssr_std[i], color='red', alpha=0.1, label='_')
        lines.append(ax2.plot( (np.arange(atrep_ssr_mean.shape[-1])+1), atrep_ssr_mean[i],   marker='.', linestyle='-', color='red',  alpha = 1 , linewidth=0.5 , label = "AtmoRep") )
        lines.append(ax2.errorbar( (np.arange(era5_ssr_mean.shape[-1]))*3  , era5_ssr_mean[i], yerr =  era5_crps_std[i], marker='P', linestyle='--', color='royalblue',  alpha = 1 , linewidth=0.5 , label = 'ERA5'))

    
      ax1.legend([lines[i][0] for i in range(2) ], [lines[i][0].get_label() for i in range(2) ], frameon = False)
      ax1.set_title( "CRPS+SSR" +' for ' + field + ' all levels')
      ax2.axhline(y=1, color='black', linestyle='--', linewidth = 0.3)
      ax2.set_xlabel('forecast time [h]')
      ax1.set_ylabel("CRPS")
      ax2.set_ylabel("SSR")
      ax2.set_xlim([0.5, 6.5])
      plt.savefig(out_dir + '/prediction_CRPS_SSR_' + field + '_all.png')
      plt.savefig(out_dir + '/prediction_CRPS_SSR_' + field + '_all.pdf')
      plt.close()

      return 0

########################################

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#############################################################

def write_csv(metric, field, data, levels, steps, label = "xxxx"):
  out = []
  for idx_lvl, level in enumerate(levels):
    temp = []
    temp.append(str(level))
    for s, step in enumerate(steps):
          temp.append(data[idx_lvl, s])
    out.append(temp)
  pd.DataFrame(out).to_csv("./{}/csv/{}_{}_{}.csv".format(out_dir, metric, field, label), header = ["level"]+ [str(i+1)+" h" for i in range(len(steps))])

########################################

def plot_persistence( source, targets, preds, tidx) :

  source_ref = source[:,-nt*token_size[0]-1]
  errs_l1 = np.zeros( (2, targets.shape[0], targets.shape[1]) )
  errs_l2 = np.zeros( (2, targets.shape[0], targets.shape[1]) )
  for ilevel in range( targets.shape[0]) :
    for i in range(targets.shape[1]) :

      errs_l2[0,ilevel,i] = np.linalg.norm( (targets[ilevel,i] - preds[ilevel,i]).flatten(), 2)
      errs_l2[1,ilevel,i] = np.linalg.norm( (targets[ilevel,i] - source_ref[ilevel]).flatten(), 2)
      
      errs_l1[0,ilevel,i] = np.linalg.norm( (targets[ilevel,i] - preds[ilevel,i]).flatten(), 1)
      errs_l1[1,ilevel,i] = np.linalg.norm( (targets[ilevel,i] - source_ref[ilevel]).flatten(), 1)
    
  errs_l2 /= (targets.shape[2] * targets.shape[3])
  errs_l1 /= (targets.shape[2] * targets.shape[3])
  colors = matplotlib.colormaps.get_cmap('plasma').resampled(errs_l2.shape[1]+1).colors

  for i in range(errs_l2.shape[1]) :
    plt.plot( np.arange(errs_l2.shape[-1])+1, errs_l2[0,i], '-x', color=colors[i])
    plt.plot( np.arange(errs_l2.shape[-1])+1, errs_l2[1,i], '--x', color=colors[i])
  plt.legend( ['prediction', 'persistence'])
  plt.title( 'Error for ' + field + ' for levels (from dark (high) to light (low))')
  plt.xlabel( 't')
  plt.ylabel( 'L_2')
  plt.savefig( dir_out + '/figures/' + 'err_persistence_' + field + '_tidx{}_L2.png'.format(tidx))
  plt.close()
  
  for i in range(errs_l1.shape[1]) :
    plt.plot( np.arange(errs_l1.shape[-1])+1, errs_l1[0,i], '-x', color=colors[i])
    plt.plot( np.arange(errs_l1.shape[-1])+1, errs_l1[1,i], '--x', color=colors[i])
  plt.legend( ['prediction', 'persistence'])
  plt.title( 'Error for ' + field + ' for levels (from dark (high) to light (low))')
  plt.xlabel( 't')
  plt.ylabel( 'L_1')
  plt.savefig( dir_out + '/figures/' + 'err_persistence_' + field + '_tidx{}_L1.png'.format(tidx))
  plt.close()