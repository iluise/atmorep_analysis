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
#  description : Code to tile up the local predictions into global images. 
#                From .dat files to a single netcdf file containing all steps
#                Dimensions: lat, lon, time, step  
#
#  license     : MIT licence
#                
####################################################################################################

import numpy as np
import glob
import json
import datetime
import os
import code
# code.interact(local=locals())
import argparse

import matplotlib
import matplotlib.pyplot as plt

# for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.linewidth'] = 0.1
import matplotlib.colors as mcolors

#for maps
import cartopy
import cartopy.crs as ccrs  #https://scitools.org.uk/cartopy/docs/latest/installing.html
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

import xarray as xr
import pandas as pd

grib_index = { 'vorticity' : 'vo', 'divergence' : 'd', 'geopotential' : 'z',
               'orography' : 'z', 'temperature': 't', 'specific_humidity' : 'q',
                'mean_top_net_long_wave_radiation_flux' : 'mtnlwrf',
                'velocity_u' : 'u', 'velocity_v': 'v', 'velocity_z' : 'w',
                'total_precip' : 'tp', 't2m' : 't_2m' }


########################################
def shape_from_str( fname) :
  shapes_str = fname.split(".")[-2].split("_s_")[-1].split("_") #remove .ext and split
  shapes = [int(i) for i in shapes_str]
  return shapes

########################################
def shape_to_str( shape) :
  ret ='{}'.format( list(shape)).replace(' ', '').replace(',','_').replace('(','s_').replace(')','')
  ret = ret.replace('[','s_').replace(']','')
  return ret

########################################
def reshape_global( arr) :
  shape = arr.shape
  return arr.transpose( [2,3,0,4,6,1,5,7]).reshape( shape[2], shape[3], 
                                                    shape[0] * shape[4] * shape[6], 
                                                    shape[1] * shape[5] * shape[7] )

########################################
def save_ims( dir_out, model_id, data, name, min_val = 1., max_val = -1.) :
  
  if not os.path.exists(dir_out + '/figures'):
    os.makedirs( dir_out + '/figures')

  cmap = matplotlib.colormaps.get_cmap('RdBu_r')

  if 1. == min_val : 
    min_val = data.min()
  if -1. == max_val : 
    max_val = data.max()
  print( 'min / max : {} / {}'.format( min_val, max_val) )

  for tidx in range( data.shape[1]) : 
    for lidx in range( data.shape[0]) :
      bname = dir_out + '/figures/fig_{}_step{:05d}_level{}_{:05d}.png'
      fname = bname.format( name, epoch, lidx, tidx)
      plt.imsave( fname, data[lidx,tidx], vmin=min_val, vmax=max_val, cmap = cmap )
 
########################################
def save_netcdf( dir_out, model_id, data, field, levels, date) :
  
  if not os.path.exists(dir_out + '/netcdf'):
    os.makedirs( dir_out + '/netcdf')

  date_range = pandas.date_range( end=date, periods=data.shape[1], freq='H')

  for ilevel, level in enumerate(levels) :

    data_xr = xr.Dataset( coords={ 'time': date_range, 
                                  'lat': np.arange( 721), 'lon': np.arange( 1440) } )
    data_xr[ grib_index[field] ] = (['time', 'lat', 'lon'], data[ilevel])

    fname_out = dir_out + '/netcdf/'
    fname_out += 'atmorep_{}_id{}_y{}_m{:02d}_ml{}.nc'.format( field, model_id, 
                                                               date.year, date.month, 
                                                               levels[ilevel])
    data_xr.to_netcdf( fname_out )
    print( 'Wrote {}'.format( fname_out))


########################################
def save_netcdf_all_steps( dir_out, model_id, data, field, levels, dates, label = "", check = False) :
  dir_out = '/p/home/jusers/luise1/juwels/atmorep/atmorep/analysis/forecast/' #HACK
  if not os.path.exists(dir_out + '/netcdf'):
    os.makedirs( dir_out + '/netcdf')
  for ilevel, level in enumerate(levels) :
    if len(data.shape) == 5: #targets preds
      data_xr = xr.Dataset( coords={'time': dates, 
                                    'step': np.arange(1, data.shape[2]+1), 
                                    'lat' : np.arange( 721), 'lon': np.arange( 1440) } )
      data_xr[ grib_index[field] ] = (['time', 'step', 'lat', 'lon'], data[ilevel])

    elif len(data.shape) == 6: #ensemble
        data_xr = xr.Dataset( coords={'time': dates, 
                                      'step': np.arange(1,data.shape[2]+1), 
                                      'member': np.arange(data.shape[3]), 
                                      'lat' : np.arange(721), 'lon': np.arange(1440) } )
        data_xr[ grib_index[field] ] = (['time', 'step', 'member', 'lat', 'lon'], data[ilevel])
    else:
      print("ERROR:: Impossible to convert to NetCDF. Please check data.")

    fname_out = dir_out + '/netcdf/'
    fname_out += 'atmorep{}_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc'.format( label, field, model_id, 
                                                                dates[0].year, dates[0].month, 
                                                                levels[ilevel])
    data_xr.to_netcdf( fname_out )
  
    if(check):
      check_atmorep(dir_out, field, model_id, dates[0], levels[ilevel])

    print( 'Wrote {}'.format( fname_out))

#sanity check
def check_atmorep(dir_out, field, model_id, date, level):
  print("Info:: inside check_atmorep.")
  fname_out = dir_out + '/netcdf/'
  fname_out +='atmorep_target_allsteps_{}_id{}_y{}_m{:02d}_ml{}.nc'.format(field, model_id, 
                                                                date.year, date.month, level)                                           
  atrep_all = xr.open_dataset(fname_out, engine = "netcdf4")

  fname_out_era5 = "/p/scratch/atmo-rep/data/era5/ml_levels/{}/{}/reanalysis_{}_y{}_m{:02d}_ml{}.grib".format( level, field, field, date.year, date.month, level) 
  era5_all = xr.open_dataset(fname_out_era5, engine="cfgrib")
  for step in np.unique(atrep_all.step.values):
    atrep = atrep_all.sel(time = date, step = step)  
    era5 = era5_all.sel(time = date+pd.DateOffset(hours=int(step)))
    assert (abs((atrep[grib_index[field]].values - era5[grib_index[field]].values).sum()) < 0.01), f"AtmoRep and ERA5 timestamps do not match, please investigate."
      
  print("Info:: check completed.")

########################################
def reshape_global( arr) :

  shape = arr.shape
  return arr.transpose( [2,3,6,0,4,7,1,5,8]).reshape( shape[2], shape[3]*shape[6],
                                                      shape[0] * shape[4] * shape[7],
                                                      shape[1] * shape[5] * shape[8] )

########################################
def reshape1( a) :
  shape = a.shape
  a = a.transpose( [0,1,2,3,6,4,7,5,8]).reshape( shape[0], shape[1], shape[2], shape[3]*shape[6],
                                                      shape[4] * shape[7],
                                                      shape[5] * shape[8] )
  return a

########################################
def reshape2( a) :
  shape = a.shape
  a = a.transpose( [2,3,0,4,1,5]).reshape( shape[2], shape[3], shape[0] * shape[4],
                                                                 shape[1] * shape[5] )
  return a

########################################
def assemble_global( arr, ap0, ap1, num_tokens_0) :

  # grid point dimensions
  ap0g = ap0 * num_tokens_0[1]
  ap1g = ap1 * num_tokens_0[2]

  if ap0 > 0 or ap1 > 0 :

    arr0 = np.expand_dims( arr[0,:,:,:,:,:], 0)
    arr0 = reshape1( arr0)
    arr0 = arr0[:,:,:,:,:-ap0g,ap1g:-ap1g]
    arr0 = reshape2( arr0)

    arr1n = arr[1:-1,:,:,:,:,:]
    arr1n = reshape1( arr1n)
    arr1n = arr1n[:,:,:,:,ap0g:-ap0g,ap1g:-ap1g]
    arr1n = reshape2( arr1n)
    
    arrn = np.expand_dims( arr[-1,:,:,:,:,:], 0)
    arrn = reshape1( arrn)
    if ap0 > 1 :
      arrn = arrn[:,:,:,:,(ap0-1)*num_tokens_0[1]:,ap1g:-ap1g]
    else :
      arrn = arrn[:,:,:,:,ap0g-1:,ap1g:-ap1g]
    arrn = reshape2( arrn)

  else :

    arr0 = np.expand_dims( arr[0,:,:,:,:,:], 0)
    arr1n = arr[1:-1,:,:,:,:,:]
    arrn = np.expand_dims( arr[-1,:,:,:,:,:], 0)

  arr = np.concatenate( (arr0, arr1n), 2)
  # take end of slice/patches since they are offset from the south pole to avoid wrap around
  arr = np.concatenate( (arr, arrn[:,:,-(721 - arr.shape[2]):]), 2)
  arr = arr[:,:,:721,:1440]

  return arr

########################################
def InvertLocalCorrectionsPerField( data, field, year, month, vlevels):
  
  corr_folder = '/p/scratch/atmo-rep/data/era5/'
  
  data_corr = np.zeros(data.shape)
  for it_vl,vl in enumerate(vlevels):
    
    bname = '/ml_levels/{}/corrections/{}/corrections_mean_var_{}_y{}_m{}_ml{}.bin'
    corr_fname = corr_folder + bname.format(vl, field, field, year, str(month).zfill(2), vl)
    corr = np.fromfile( corr_fname, dtype=np.float32).reshape((721, 1440, 2)).transpose( (2,0,1)) 
    
    data_corr[it_vl] = data[it_vl] * corr[1] + corr[0]

  return data_corr

########################################
def InvertGlobalCorrectionsPerField( data, field, year, month, vlevels):

  corr_folder = '/p/scratch/atmo-rep/data/era5/'
  data_corr = np.zeros(data.shape)
  
  assert len(vlevels) == data.shape[0]

  for it_vl, vl in enumerate(vlevels):
    
    corr_fname = corr_folder+'ml_levels/{}/corrections/global_corrections_mean_var_{}_ml{}.bin'.format(vl, field, vl)
    x = np.fromfile(corr_fname, dtype=np.float32)
    corr = np.reshape(x, (-1, 4))

    corr_data_ym = list(corr[np.where(np.logical_and(corr[:, 0] == year, corr[:, 1] == month))][0, 2::])
    mean = corr_data_ym[0]
    std  = corr_data_ym[1]

    data_corr[it_vl] = data[it_vl] * std + mean

  return data_corr

####################################################################################################

 # Parse the command-line arguments                     
parser = argparse.ArgumentParser(description="Program to invert normalisations.")
parser.add_argument("--model-id", "-m", dest="model_id", type=str, required=True, 
                      help="model ID without \"id\".")

base_dir = '/p/scratch/deepacf/results/'
args = parser.parse_args()
model_id = args.model_id

num_batches_per_global = 20
bname = base_dir + '/id' + model_id + '/'

with open( bname + '/model_id'+ model_id + '.json') as json_file:
  config = json.load(json_file)

# output
dir_out = bname + 'analysis'
if not os.path.exists(dir_out):
  os.makedirs( dir_out)
  os.makedirs( dir_out + '/figures')
print( 'Output for id = \'{}\' written to: {}'.format( model_id, dir_out))

if 'forecast_num_tokens' in config :
  nt = config['forecast_num_tokens']
else :
  nt = 1
num_ens = config['net_tail_num_nets']
print( 'overlap: {}'.format( config['token_overlap'])) 
# set_global uses fields[0] so also use it here to determine and cut out apron region
num_tokens_0 = config['fields'][0][4]
ap0 = int(config['token_overlap'][0] / 2)
ap1 = int(config['token_overlap'][1] / 2)
print( 'ap0 / ap1: {} / {}'.format( ap0, ap1)) 

nbg = num_batches_per_global

for ifield, field_info in enumerate(config['fields']) :
  tidx = 0
  more_tsteps = True
  stddevs = []
  preds_all = []
  targets_all = []
  ens_all = []
  all_dates = []
  field = field_info[0]
  if field == 'total_precip' or field == 'velocity_z':
      continue
  while more_tsteps :
    print( 'Processing {} at tidx = {}.'.format( field, tidx) )
    levels = field_info[2]
    num_tokens = field_info[3]
    token_size = field_info[4]
    local_corrections = False
    if len(field_info) > 6 :
      if 'local' == field_info[6] :
        local_corrections = True
        inv_corr = InvertLocalCorrectionsPerField
      else :
        inv_corr = InvertGlobalCorrectionsPerField

    cat = np.concatenate
    f32 = np.float32

    for epoch in range( 1) : 

      # token infos
      fname = bname + '*{:05d}_'.format(epoch) + '*_token_infos_' + field + '*.dat'
      fname_token_infos = sorted(glob.glob( fname))
      if not (len(fname_token_infos) >= (tidx+2)*nbg) :
        more_tsteps = False

      fname_token_infos = fname_token_infos[tidx*nbg : (tidx+1)*nbg]
      token_infos = cat( [np.expand_dims( np.fromfile(f, dtype=f32).reshape( shape_from_str(f)), 0) 
                                                                  for f in fname_token_infos], 0)
      # extract year and month; global forecast so can be deduced from first token
      tok_info_0 = token_infos[0,0,0]
      date = datetime.datetime.strptime('{} {} {}'.format( int(tok_info_0[0]),
                                                          int(tok_info_0[1]+2), 
                                                          # int(tok_info_0[2])), '%Y %j %H') + pd.DateOffset(hours=6) #is good for 3h
                                                          int(tok_info_0[2])), '%Y %j %H') + pd.DateOffset(hours=3) #is good for 6h 
      all_dates.append(date)
      # source
      fname_sources = sorted(glob.glob(bname + '*{:05d}_'.format(epoch)+'*_source_'+field+'*.dat'))
      fname_sources = fname_sources[tidx*nbg : (tidx+1)*nbg]
      source = cat( [np.expand_dims( np.fromfile(f, dtype=f32).reshape( shape_from_str(f)), 0) 
                                                                        for f in fname_sources], 0)
      source = assemble_global( source, ap0, ap1, num_tokens_0)
      source = inv_corr( source, field, date.year, date.month, levels)

      # predictions
      fname_preds  = sorted(glob.glob( bname + '*{:05d}_'.format(epoch) + '*_preds_'+field+'*.dat'))
      fname_preds = fname_preds[tidx*nbg : (tidx+1)*nbg]
      preds = cat( [np.expand_dims(np.fromfile( f, dtype=np.float32).reshape( shape_from_str(f)), 0)
                                                                              for f in fname_preds])
      rshape = [preds.shape[0], len(levels), -1, nt] + num_tokens[1:] + token_size
      preds = preds.reshape( rshape).transpose( 0, 2, 1, 3, 4, 5, 6, 7, 8)
      preds = assemble_global( preds, ap0, ap1, num_tokens_0)
      preds = inv_corr( preds, field, date.year, date.month, levels)

      # targets
      fname_targets = sorted(glob.glob( bname + '*{:05d}_'.format(epoch)+'*_target_'+field+'*.dat'))
      fname_targets = fname_targets[tidx*nbg : (tidx+1)*nbg]
      targets = cat( [np.expand_dims(np.fromfile( f, dtype=np.float32).reshape( shape_from_str(f)), 0)
                                                                              for f in fname_targets])
      rshape = [targets.shape[0], len(levels), -1, nt] + num_tokens[1:] + token_size
      targets = targets.reshape( rshape).transpose( 0, 2, 1, 3, 4, 5, 6, 7, 8)
      targets = assemble_global( targets, ap0, ap1, num_tokens_0)
      targets = inv_corr( targets, field, date.year, date.month, levels)

      # ensemble
      fname_ens = sorted(glob.glob( bname + '*{:05d}_'.format(epoch)+'*_ensembles_'+field+'*.dat'))
      fname_ens = fname_ens[tidx*nbg : (tidx+1)*nbg]
      ens = cat( [np.expand_dims(np.fromfile( f, dtype=np.float32).reshape( shape_from_str(f)), 0)
                                                                              for f in fname_ens])
      rshape = [ens.shape[0], len(levels), -1, nt] + num_tokens[1:] + [num_ens] + token_size
      ens = ens.reshape( rshape).transpose( 6, 0, 2, 1, 3, 4, 5, 7, 8, 9)
      ens_global = []
      for i, _ in enumerate(ens) :
        temp = assemble_global( ens[i], ap0, ap1, num_tokens_0)
        temp = inv_corr( temp, field, date.year, date.month, levels) 
        ens_global.append( np.expand_dims( temp, 0))
      # move ensemble dimension to third dimension
      ens = cat( ens_global, 0)
      ens = ens.transpose( 1, 2, 0, 3, 4)

      preds_all.append(preds)
      targets_all.append(targets)
      ens_all.append(ens)
      stddev = np.std( ens, axis=2)
     
      stddevs.append( stddev)

    tidx += 1
    # if tidx > 6: #for R&D
    #   break

  if(len(all_dates) > 0):
    print("Dates:",  all_dates[0], '-',  all_dates[-1])
    preds_all   = np.array(preds_all, dtype = np.float32).swapaxes(0, 1)
    targets_all = np.array(targets_all, dtype = np.float32).swapaxes(0,1)
    ens_all = np.array(ens_all, dtype = np.float32).swapaxes(0,1)
    save_netcdf_all_steps( dir_out, model_id, targets_all, field, levels, all_dates, "_target", check = True)
    save_netcdf_all_steps( dir_out, model_id, ens_all, field, levels, all_dates, "_ensemble")
    save_netcdf_all_steps( dir_out, model_id, preds_all, field, levels, all_dates, "")
