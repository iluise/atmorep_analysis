"""
Collection of useful functions used by the different analysis scripts of the downstream applications
Credits to: AtmoRep collaboration
Date: Last revision - August 2023

"""

import xarray as xr
import numpy as np
import pandas as pd
from itertools import product
import datetime as dt
from calendar import monthrange
import datetime
import copy
import code
from typing import Any, List, Union
import pathlib

grib_index = { 'vorticity' : 'vo', 'divergence' : 'd', 'geopotential' : 'z',
               'orography' : 'z', 'temperature': 't', 'specific_humidity' : 'q',
                'velocity_u' : 'u', 'velocity_v': 'v', 'velocity_z' : 'w',
                'total_precip' : 'tp', 't2m' : 't_2m', 'radar_precip': 'yw_hourly' }

##########################################

units = { 'vorticity' : 's$^{-1}$', 'divergence' : 's$^{-1}$', 'geopotential' : 'm$^2$ s$^{-2}$',
          'temperature': 'K', 'specific_humidity' : 'kg kg$^{-1}$',
          'velocity_u' : 'm s$^{-1}$', 'velocity_v': 'm s$^{-1}$', 'velocity_z' : '???',
          'total_precip' : 'm', 't2m' : 'K', 'radar_precip' : 'mm/h'}

##########################################

field_full_name = {"velocity_u": "U component",
                   "velocity_v": "V component",
                  "temperature":  "Temperature", 
                   "specific_humidity": "Specific Humidity", 
                   "divergence" : "Divergence", 
                   "vorticity":   "Vorticity"}

##########################################

def get_pl_level(ml):
  levels = {137 : 1000, 
            123 : 925, 
            114 : 850, 
            105 : 700, 
            96  : 500}
  assert ml in levels.keys(), f"ml level {ml} not supported. please choose among: {levels.keys()}."
  return levels[ml]

##########################################
  
def get_units(field):
  return units[field]

##########################################

def pad(array, target_shape):
    return np.pad(
        array,
        [(0, target_shape[i] - array.shape[i]) for i in range(len(array.shape))],
        "constant",
    )

##########################################

def shape_from_str( fname) :
  shapes_str = fname.replace("_phys.dat", ".dat").split(".")[-2].split("_s_")[-1].split("_") #remove .ext and split
  shapes = [int(i) for i in shapes_str]
  return shapes

##########################################

def shape_to_str( shape) :
  ret ='{}'.format( list(shape)).replace(' ', '').replace(',','_').replace('(','s_').replace(')','')
  ret = ret.replace('[','s_').replace(']','')
  return ret

##########################################

def remove_redundant_axes(data, axes = []):

  data_coord = data
  for ax in axes:
    data_coord = np.unique(data_coord, axis=ax) #remove redundant lat
    
  data_coord = np.squeeze(data_coord, axis=axes)

  return data_coord

##########################################

def get_era5_forecasts(grib_file, forecast_steps = [0, 12]):
    """
    Auxiliary function for handling ERA5-forecasts.
    Opens a grib file, originally including precipitation forecasts (dimension step).
    Inputs: 
        grib_file: path to the grib file 
    Returns:
        ds_era5: xarray with the era5 (precipitation) with adjusted dimensions
    """
    ds_era5 = xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs={'indexpath': ''})#, backend_kwargs={'time_dims':('valid_time','indexing_time')})
    ds_era5 = ds_era5.isel({"step": slice(forecast_steps[0], forecast_steps[1] + 1)})
    # re-organize data, so that forecast-dimension is boiled to continuous time-dimension
    # From: https://stackoverflow.com/questions/67342119/xarray-merge-separate-day-and-hour-dimensions-into-one-time-dimension-in-python
    ds_era5 = ds_era5.assign_coords({"valid_time": ds_era5.time + ds_era5.step})    
    ds_era5 = ds_era5.stack(datetime=("time", "step"))
    ds_era5 = ds_era5.drop_vars("datetime").rename_dims({"datetime": "time"})\
              .rename_vars({"valid_time": "time"}).set_index(time="time")
    ds_era5 = ds_era5.assign_coords({"time": ds_era5["time"]})
    
    return ds_era5

def accumulate(da, interval='3H', skipna: bool = False, time_dim: str = "time", acc_dim: str = None, bins: List = None):
    """ 
    Simple function to accumulate a variable over n hours and store the information 
    at the end of the time period. 
    Input: 
        datset: xarray to be accumulated, needs an time dimension.
        interval: string or integer, accumulation interval
        skipna: flag to indicate if missing values are skipped. 
    
    """ 
    if isinstance(interval, str):
        intrvl = int(interval.strip('H'))
    elif isinstance(interval, int):
        intrvl = interval
        interval = str(interval) + 'H'
    else: 
        print(interval)
        raise ValueError("interval should be a string of the format [n hours]H or an integer.")
    if not 24 % intrvl == 0:
        raise ValueError("interval must be a divider of 24 hours!")
    
    if acc_dim is None:
        da_acc = da.resample({time_dim: interval}, closed="right", label='right', skipna=skipna).sum(skipna=skipna) 
    else:
        da_acc = da.groupby_bins(acc_dim, bins, right=False, include_lowest=True)
        da_acc= da_acc.sum(skipna=skipna)
    return da_acc

def season(mmm='JJA', year=2018):
    """
    Takes in a string with the first three letters of the months of the corresponding season on the North Hemisphere and returns the date range.
    Inputs: 
        mmm: String (e.g. DJF, MAM, JJA, SON)
        year: year to evaluate
    """ 
    if mmm=='DJF':
        m_s = dt.datetime(year-1, 12, 1)
        if year % 4 == 0:
            m_e = dt.datetime(year, 2, 29)
        else:
            m_e = dt.datetime(year, 2, 28)
    elif mmm=='MAM':
        m_s = dt.datetime(year, 3, 1)
        m_e = dt.datetime(year, 5, 31)
    elif mmm=='JJA':
        m_s = dt.datetime(year, 6, 1)
        m_e = dt.datetime(year, 8, 31)
    elif mmm=='SON':
        m_s = dt.datetime(year, 9, 1)
        m_e = dt.datetime(year, 11, 30)
    else:
        raise ValueError("The input for season() has to be DJF, MAM, JJA or SON")
    
    date_start = m_s.strftime('%Y-%m-%d') 
    date_end = m_e.strftime('%Y-%m-%d') 
    return date_start, date_end

def to_list(obj: Any) -> List:
    """
    Method from MLAIR!
    Transform given object to list if obj is not already a list. Sets are also transformed to a list
    :param obj: object to transform to list
    :return: list containing obj, or obj itself (if obj was already a list)
    """
    if isinstance(obj, (set, tuple)):
        obj = list(obj)
    elif not isinstance(obj, list):
        obj = [obj]
    return obj

def doy_to_mo(day_of_year: int, year: int):
    """
    Converts day of year to year-month datetime object (e.g. in_day = 2, year = 2017 yields January 2017).
    :param day_of_year: day of year
    :param year: corresponding year (to reflect leap years)
    :return year-month datetime object
    """
    date_month = pd.to_datetime(year * 1000 + day_of_year, format='%Y%j')
    return date_month

##############################################################

def compute_climatological_mean(field, month, vlevel):
  print("Info:: Calculating local climatological mean.")
  
  labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
  month_label = labels[month-1]
  
  lvl_type = 'ml'
  if vlevel > 137:
    lvl_type = 'pl'

  mean = np.zeros([721, 1440])

  if lvl_type == 'ml':
    years = range(1979, 2021) 
    corr = np.zeros([len(years), 2, 721, 1440], dtype = np.float32) #TO-DO: avoid hardcoding here
    corr_folder = '/p/scratch/atmo-rep/data/era5/'

    for it_y, y in enumerate(years):
      corr_fname = corr_folder+'/{}_levels/{}/corrections/{}/corrections_mean_var_{}_y{}_m{}_{}{}.bin'.format(lvl_type, vlevel, field, field, y, str(month).zfill(2), lvl_type, vlevel)
      x = np.fromfile(corr_fname, dtype=np.float32)
      x = np.reshape(x, (721, 1440, 2))
      x = np.transpose(x, (2,0,1))  
      corr[it_y] = np.array(x, dtype = np.float32)
      
    mean = np.mean(corr[ :, 0, :, :], axis = 0).squeeze()
    
  else:
    ds = xr.open_dataset("/p/scratch/atmo-rep/data/era5/pl_levels/monthly_averages_pl_levels.grib", engine = "cfgrib")
    data = ds.groupby(ds.time.dt.month)[month].sel(isobaricInhPa = vlevel)
    print(data)
    mean = np.mean(data[grib_index[field]].values, axis = 0)
    print(mean.shape)

  mean.tofile('/p/scratch/atmo-rep/data/era5/climatological_means/local_average_values_' + field + '_'+ str(lvl_type) + str(vlevel) + '_'+ month_label +'.dat' )
  
  return mean 

##############################################################

def get_climatological_mean(field, month, vlevel):
  #TO-DO: Make it more solid when new levels will come
  lvl_type = 'ml'
  if vlevel > 137:
    lvl_type = 'pl'

  labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
  month_label = labels[month-1]
 
  mean  = np.zeros([721, 1440], dtype = np.float32)
  filename = '/p/scratch/atmo-rep/data/era5/climatological_means/local_average_values_' + field + '_'+ str(lvl_type) + str(vlevel) + '_'+ month_label + '.dat'
  
  if pathlib.Path(filename).is_file():
    mean = np.fromfile(filename,  dtype = np.float32 ).reshape(721, 1440) 
  return mean 

def get_element_string(string_list, substring: str):
  """
  Get first element from list of strings containing substring.
  :param string_list: the list of strings
  :param substring: the substring to search for
  :return: the desired element
  """
  element = [s for _, s in enumerate(string_list) if substring in s][0]

  return element
