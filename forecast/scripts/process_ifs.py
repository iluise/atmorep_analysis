# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#                         2023 Otto-von-Guericke-University, Magdeburg (OvGU)
#                         2023 CERN IT Departement
#
# SPDX-License-Identifier: MIT

"""
Script to process IFS forecast provided grib-files per IFS-run (00 and 12 UTC) into monthly netCDF-files for each variable/field and level.
"""
__author__ = "Christian Lessig, Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2023-07-28"
__update__ = "2023-07-28"

import os
import xarray as xr
import numpy as np
from calendar import monthrange
import pandas as pd


def days_in_month(year, month):
  """
  Provides number of days in month of specific year.
  :param year: year 
  :param month: month for which number of days is requested
  :return number of days in particular year-month
  """
  return monthrange(year, month)[1]

def add_inittime_dim(ds, offset: int = 1):
  """
  Add initial time as singleton dimension to xarray Dataset (named time!).
  Furthermore, valid_time is renamed to steps which also becomes an integer array 
  :param ds: dataset with valid_time-dimension 
  :param offset: offset between first valid_time-value and initial time
  :return ds: dataset with additional singleton time-dimension
  """
  
  nsteps = len(ds["valid_time"])
  init_time = pd.to_datetime(ds["valid_time"][0].values) - pd.Timedelta(offset, "h")

  ds = ds.expand_dims({"time": [init_time]}, 0)
  ds = ds.rename({"valid_time": "step"})
  ds["step"] = range(offset, offset + nsteps)

  return ds

# parameters
basedir_in = "/p/largedata/slmet/slmet111/met_data/ecmwf/oper/"
basedir_out = "/p/scratch/atmo-rep/data/IFS/"

years = [2020]

fields = ['velocity_u', 'velocity_v', 'velocity_z', 'specific_humidity', 'temperature'] #, 'vorticity', 'divergence']
levels = [96, 105, 114, 123, 137]
vtype = 'ml'

# auxiliary dictionary
grib_index = { 'vorticity' : 'vo', 'divergence' : 'd', 'geopotential' : 'z',
               'orography' : 'z', 'temperature': 't', 'specific_humidity' : 'q', 
                'mean_top_net_long_wave_radiation_flux' : 'mtnlwrf',
                'velocity_u' : 'u', 'velocity_v': 'v', 'velocity_z' : 'w',
                'total_precip' : 'tp', 't2m' : 't_2m' }

# fields = ['total_precip']
# levels = [0]
# vtype = 'sf'

for year in years:
#  for month in range(1, 13):
  for month in range( 11, 13): #  for testing

    print(f"Process data for {year:d}-{month:02d}...")

    # get current data directory 
    indir_now = os.path.join(basedir_in, f"{year}", f"{month:02d}")
    if not os.path.isdir(indir_now):
      raise NotADirectoryError(f"Could not find expected input directory '{indir_now}'")
    # re-initialize flag for first dataset 
    first = True

    print("Start collecting raw data...")
    for day in range(1, days_in_month( year, month)+1):
    #for day in range(1, 2): # for testing
      for hh in [0, 12]:
        fname = os.path.join(indir_now, f"{year}{month:02d}{day:02d}{hh:02d}_{vtype}.grb")
        print(f"Reading {fname}...")

        ds_now = xr.open_dataset( fname, engine='cfgrib',
                                  backend_kwargs={'time_dims':('valid_time','indexing_time'),'indexpath':''})
        ds_now = add_inittime_dim(ds_now, offset=1)
        
        if first:
          ds_all = ds_now.copy()
          first = False
        else:
          ds_all = xr.concat([ds_all, ds_now], dim="time")
      
    print("Data successfully read. Start to generate netCDF-files...")
    # rename hybrid-dimension
    ds_all = ds_all.rename({"hybrid": "vlevel"})

    for field in fields:
      for lvl in levels:
        outdir = os.path.join(basedir_out, field,  f"ml{lvl}")
        if not os.path.isdir(outdir):
          print(f"Create output-directory '{outdir}'...")
          os.makedirs(outdir, exist_ok=True)

        fname_out = os.path.join(outdir, f"ifs_forecast_{field}_y{year:d}_m{month:02d}_ml{lvl:d}.nc")
        ds_all[grib_index[field]].sel({"vlevel": lvl}).to_netcdf(fname_out)
        print(f"Successfully wrote netCDF-file '{fname_out}'.")

    print(f"Data successfully processed for {year:d}-{month:02d}.")
                             
                        


