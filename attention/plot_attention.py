import numpy as np
import glob
import json
import datetime
import os
import code
import zarr
import xarray as xr
import pandas

# for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.linewidth'] = 0.1
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#for maps
import cartopy
import cartopy.crs as ccrs  #https://scitools.org.uk/cartopy/docs/latest/installing.html
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

from attention_utils import *
from utils.utils import shape_from_str, grib_index

base_dir     = '/p/scratch/atmo-rep/results/'
model_id     =  '1zq3f5gf' #'8x5dnvc9' #'bkkrobb9' #'xsbasstk' #'q7yr6hrj'
out_folder   = "./attention_plots/"
tag = "" #if you want to add a particular description to your plots

#values:
layer = 0 
head  = 0

#reference coordinates:
# coordinates of the token to which the attention is referred to for each local patch. 
rf_lvl  = 0
rf_time = 0
rf_lat  = 3
rf_lon  = 6

##################################  
def plot_attention(field, filename, source, idxs_nb, toksize = [12, 6, 12]):

  ########## LOAD ALL HEADS ##############
  attn = load_attention(filename, field)
  print("Info:: Start plotting.")
  print(attn.info)
  temp = get_layer(attn, batch = idxs_nb, layer = layer)
  datetime = temp["datetime"]
  ml = temp["ml"]

  ds_o = xr.Dataset( coords={ 'ml': ml,  
                              'datetime': np.unique(datetime),
                              'lat' : np.linspace( 0., 180., num=81, endpoint=True),
                              #'lat' : np.linspace( -90., 90., num=81, endpoint=True),
                              'lon' : np.linspace( 0., 360., num=160, endpoint=False) } )

  ds_o[grib_index[field]] = (['ml', 'datetime', 'lat', 'lon'], np.zeros( ( len(ml), toksize[0], 81, 160)))

  #TO-DO: skipping the last batch since it's problematic
  for b1 in range(idxs_nb, idxs_nb + datetime.shape[0]-1):
    temp = get_head(attn, batch = b1, layer = layer, head = head)
    lat, lon = get_lat_lon(attn, batch = b1, layer = layer)
    temp = temp.reshape(*temp.shape[:3], toksize[1], toksize[2], *temp.shape[-3:-1], toksize[1], toksize[2])
    for b in range(datetime.shape[0]):
      ds_o[grib_index[field]].loc[ dict(
                           datetime = datetime[b], 
                           lat=lat[b],
                           lon=lon[b]) ] = temp[b, rf_lvl, rf_time, rf_lat, rf_lon]

  print(ds_o)
  cmap='PuBuGn'
  #  ########### PLOT x TIME ###############
  data = ds_o[grib_index[field]].isel(ml = rf_lvl)
  vmin, vmax = data.min(), data.max()
  for k in range(datetime.shape[1]):
    fig = plt.figure( figsize=(10,5), dpi=300)
    ax = plt.axes( projection=cartopy.crs.Robinson( central_longitude=0.))
    ax.add_feature( cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
    ax.set_global()
    date = ds_o['datetime'].values[k].astype('datetime64[m]')
    ax.set_title(f'{field} : {date}')
    data.isel(datetime = k).plot.imshow(cmap=cmap, vmin=vmin, vmax=vmax)
    #    im = ax.imshow( np.flip(ds_o[grib_index[field]].values[rf_lvl, k], 0), cmap=cmap, vmin=vmin, vmax=vmax,
    im = ax.imshow( data.isel(datetime = k), cmap=cmap, vmin=vmin, vmax=vmax,
                    transform=cartopy.crs.PlateCarree( central_longitude=180.))
    axins = inset_axes( ax, width="80%", height="5%", loc='lower center', borderpad=-2 )
    fig.colorbar( im, cax=axins, orientation="horizontal")
    plt.savefig( f'example_{field}_time_{k:03d}.png')
    plt.close()

  #  ########### PLOT x LEVEL ###############
  print("print per level")
  data = ds_o[grib_index[field]].isel(datetime = rf_time)
  vmin, vmax = data.min(), data.max()
  for k in range(len(data["ml"])):
    fig = plt.figure( figsize=(10,5), dpi=300)
    ax = plt.axes( projection=cartopy.crs.Robinson( central_longitude=0.))
    ax.add_feature( cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
    ax.set_global()
    level = ds_o['ml'].values[k]
    date = ds_o['datetime'][rf_time].astype('datetime64[m]')
    ax.set_title(f'{field} : level {level} - {date} ')
    data.isel(ml = k).plot.imshow(cmap=cmap, vmin=vmin, vmax=vmax)
    #    im = ax.imshow( np.flip(ds_o[grib_index[field]].values[rf_lvl, k], 0), cmap=cmap, vmin=vmin, vmax=vmax,
    im = ax.imshow( data.isel(ml = k), cmap=cmap, vmin=vmin, vmax=vmax,
                    transform=cartopy.crs.PlateCarree( central_longitude=180.))
    axins = inset_axes( ax, width="80%", height="5%", loc='lower center', borderpad=-2 )
    fig.colorbar( im, cax=axins, orientation="horizontal")
    plt.savefig( f'example_{field}_lvl_{k:03d}.png')
    plt.close()
  
#  ########### PLOT x BATCH ###############
#  label = f"lyr{layer}_h{head}_lvl{level}_t{time}"
#  if tag != "":
#    label = tag+ "_" + label
#  plot_vs_batch_number(out_folder, field, source, temp, label)
#  
#  ############# PLOT x LEVEL #############
#  plot_vs_vertical_level(field, source, attn, label, mode)
#  
#  ############ PLOT x TIME STEP ##########
#  plot_vs_time_step(field, source, attn, label, mode)
#  
#  ############## PLOT x HEADS ############
#  plot_vs_head_number(field, source, attn, label, mode)

  return 0

  
##########################################

def main():
  bname = base_dir + '/id' + model_id + '/'

  with open( bname + '/model_id'+ model_id + '.json') as json_file:
    config = json.load(json_file)
  
#  print(config)
  num_heads = config["encoder_num_heads"]
  
  # output
  if not os.path.exists(out_folder):
    os.makedirs( out_folder)
  print( 'Info:: Output for id = \'{}\' written to: {}'.format( model_id, out_folder))
  more_tsteps = True
  tidx = 0

  cat = np.concatenate
  f32 = np.float32

  for ifield, field_info in enumerate(config['fields']) :

    field = field_info[0]

    if field == "temperature" or field == "total_precip":
      continue
    
    print( 'Info:: Processing {}.'.format( field ) )

    # source
#   fname_source = glob.glob(base_dir +'/id'+ model_id +'/*source*.zarr')[0]
#   store = zarr.ZipStore(fname_source, mode='r')
#   root_source = zarr.open_group(store)[field]
#   source = []
#   for name, sample in root_source.groups():
#     source_temp = sample.data[:]
#     source.append(source_temp)
#   source = np.array(source)
#   print( 'Info:: source : {}'.format( source.shape ) )
#
#   rng = np.random.default_rng()
#   idxs_nb = rng.permutation( source.shape[0] )[:num_samples]
#   source = source[idxs_nb, :]
#   print(source.shape)
#
    source = None
    idx_nb = 0 #list(range(10))
    #attention file
    filename = glob.glob(base_dir +'/id'+ model_id +'/*attention*.zarr')[0]
    plot_attention(field, filename, source, idx_nb)
    
if __name__ == '__main__':
  main()
