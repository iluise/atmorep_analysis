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
#  description : Code to calculate all climatological means for the ACC score. 
#
#  license     : MIT licence
#                
####################################################################################################

import zarr
import xarray as xr
import numpy as np
import pandas as pd

def compute_climatological_mean(out_path, in_path = "/gpfs/scratch/ehpc03/era5_y2010_2021_res025_chunk8.zarr/"):
    store = zarr.DirectoryStore(in_path)
    root = zarr.group(store)
    lats = root.lats[:]
    lons = root.lons[:]
    fields = root.attrs["fields"]
    levels = root.attrs["levels"]
    data = root["normalization/norm"][:, 0].mean(axis = 0)
    # time = pd.to_datetime(root.time)
    # time = pd.Series(time)
    # time = time.dt.dayofyear
   
    # data_all = []
    # len_time= range(1,4) #np.unique(time)
    # for day in len_time:
        # print("day:", day)
        # idxs = np.where(time == day)[0]
        # data = root["data"][idxs].mean(axis = 0)
        # data_all.append(data)
    # breakpoint()
    # data_all = np.stack(data_all)
    da = xr.Dataset(coords={'ml': levels, 'lat': lats, 'lon' :lons})
    for fidx, field in enumerate(fields):
         # ds_o['field'] = (['ml', 'datetime', 'lat', 'lon'], np.zeros( ( nlevels, 6, 721, 1440)))
        da[field] = (["ml", "lat", "lon"], data[fidx])
    
    #da = xr.DataArray(data, coords=[fields, levels, lats, lons], dims = ["field", "ml", "lat", "lon"])    
    da.to_zarr(out_path)


compute_climatological_mean("climatological_mean1.zarr")

# def compute_climatological_mean(field, month, vlevel):
 
#     print("Info:: Calculating local climatological mean.")
    
#     labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
#     month_label = labels[month-1]

#     mean = np.zeros([721, 1440])

#     years = range(2010, 2022) 
#     corr = np.zeros([len(years), 2, 721, 1440], dtype = np.float32) #TO-DO: avoid hardcoding here
#     corr_folder = '/gpfs/scratch/ehpc03/data/normalization/'

#     for it_y, y in enumerate(years):
#         corr_fname = corr_folder+'/{}/normalization_mean_var_{}_y{}_m{}_ml{}.bin'.format(field, field, y, str(month).zfill(2), vlevel)
#         x = np.fromfile(corr_fname, dtype=np.float32)
#         x = np.reshape(x, (721, 1440, 2))
#         x = np.transpose(x, (2,0,1))  
#         corr[it_y] = np.array(x, dtype = np.float32)
        
#     mean = np.mean(corr[ :, 0, :, :], axis = 0).squeeze()

#     mean.tofile('climatological_means/local_average_values_' + field + '_ml' + str(vlevel) + '_'+ month_label +'.dat' )

# field = 'velocity_u'
# month = 1
# lvl = 137
# compute_climatological_mean(field, month, lvl)
    #return mean 


