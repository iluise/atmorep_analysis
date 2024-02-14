# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#                         2023 Otto-von-Guericke-University, Magdeburg (OvGU)
#                         2023 CERN IT Departement
#
# SPDX-License-Identifier: MIT

"""
Script to process PanguWeather grib files that provide six hourly forecast initialized each hour.
Either processes hourly provided forecasts until day 5 of the month (ifs_like: False)
or only processes forecasts initialized at 00 and 12 UTC, but for all days of the month.
"""
__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2023-07-31"
__update__ = "2023-07-31"

import os
import xarray as xr
import cfgrib
import numpy as np
from calendar import monthrange
import pandas as pd


def days_in_month(year, month) :
  """Days in month of specific year"""
  return monthrange(year, month)[1]


# parameters
dir_in = "/p/scratch/atmo-rep/scratch/"
dir_out = os.path.join(dir_in, "netcdf")

years = [2020]
month = [6]                # set month=None to process all months of a year
fields = ['velocity_u', 'velocity_v', 'specific_humidity', 'temperature']
vtype = "isobaricInhPa"
plevels = [1000., 925., 850., 700., 500.]                             # pressure-levels in hPa
like_ifs = False

# auxiliary dictionary
grib_index = { 'vorticity' : 'vo', 'divergence' : 'd', 'geopotential' : 'z',
               'orography' : 'z', 'temperature': 't', 'specific_humidity' : 'q', 
                'mean_top_net_long_wave_radiation_flux' : 'mtnlwrf',
                'velocity_u' : 'u', 'velocity_v': 'v', 'velocity_z' : 'w',
                'total_precip' : 'tp', 't2m' : 't_2m' }

if like_ifs:
  hh_list = [0, 12]
  add_string = "_00+12runs"
else:
  hh_list = list(range(0, 24))
  day_list = range(1, 5)
  add_string = ""

# translate varable names
varlist = [grib_index[var] for var in fields]
print(month)
if month is None:
  months = range(1, 13)
else:
  months = list(month)

for year in years:
  for month in months:
    print(f"Process data for {year:d}-{month:02d}...")

    # re-initialize flag for first dataset 
    first = True

    print("Start collecting raw data...")
    if like_ifs:
      day_list = range(1, days_in_month(year, month)+1)
    for day in day_list:
      for hh in hh_list:
        fname = os.path.join(dir_in, f"output_panguweather_6h_forecast_{year}{month:02d}{day:02d}_{hh:02d}00.grib")
        if not os.path.isfile(fname):
          raise FileNotFoundError(f"Could not find requested PanguWeather-file {fname}")
        print(f"Reading {fname}...")

        ds_now = xr.open_dataset(fname, backend_kwargs={'filter_by_keys': {'typeOfLevel': vtype}})
        if vtype == "isobaricInhPa":
          ds_now = ds_now[varlist].sel({"isobaricInhPa": plevels})
        else:
          ds_now = ds_now[varlist]
        
        if first:
          ds_all = ds_now.copy()
          first = False
        else:
          ds_all = xr.concat([ds_all, ds_now], dim="time")
      
    print("Data successfully read. Start to generate netCDF-file...")
    print(ds_all)
    fname_out = os.path.join(dir_out, f"output_panguweather_6h_forecast_{year}{month:02d}{add_string}.nc")
    ds_all.to_netcdf(fname_out)

    print(f"Data successfully written to {fname_out}...")
                             
                        


