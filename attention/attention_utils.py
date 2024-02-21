import numpy as np
import glob
import json
import datetime
import os
import code
import xarray as xr
import pandas
import zarr
import pdb

# for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.linewidth'] = 0.1
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

#for maps
import cartopy
import cartopy.crs as ccrs  #https://scitools.org.uk/cartopy/docs/latest/installing.html
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

from utils.utils import shape_from_str, grib_index
from utils.read_atmorep_data import HandleAtmoRepData
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class HandleAtmoRepAttention(object):
  """
    Handle AtmoRep attention data
    TODO:
    - fix latitude issue in last batch item 
    """
  def __init__(self, field: str, out_folder: str, tag: str, rf_dict : dict, verbose = False, cmap = 'PuBuGn'):

    self.field      = field
    self.out_folder = out_folder
    self.tag        = tag
    self.rf_lvl     = rf_dict["lvl"]
    self.rf_time    = rf_dict["time"]
    self.rf_lat     = rf_dict["lat"]
    self.rf_lon     = rf_dict["lon"]
    self.verbose    = verbose
    self.cmap       = cmap
  
  def inspect_attention(attn):
    print(type(attn))
    print(attn.shape)

  def set_tag(self, tag):
    self.tag = tag
    
  #################################
    
  def load_attention(self, filename):
    store = zarr.ZipStore(filename, mode='r')
    ds = zarr.open_group(store)[f'{self.field}/'] 
    print("Info:: Attention maps loaded successfully.")
    return ds

  def get_batch(self, ds, batch):
    return ds[ f'batch={batch:05d}']
    
  def get_layer(self, ds, batch, layer):
    return ds[ f'batch={batch:05d}/layer={layer:05d}' ]
    
  def get_head(self, ds, batch, layer, head):
    return ds[ f'batch={batch:05d}/layer={layer:05d}/heads/{head}' ][:]
    
  def get_lat_lon(self, ds, batch, layer):
    lat = ds[ f'batch={batch:05d}/layer={layer:05d}'].lat[:]
    lon = ds[ f'batch={batch:05d}/layer={layer:05d}'].lon[:]
    return lat, lon

  #################################

  def add_plot(self, axs, source_img, attn_img, idx, nrows, ncols, title):
    
    if source_img != None:
      # Determine the row and column indices for the subplots.
      row = idx // ncols
      col = idx % ncols
      
      # Plotting the source image on the first and third rows.
      axs[row*2, col].imshow(source_img, cmap='PuBuGn')
      axs[row*2, col].set_title(f'Source Image {idx+1}')
      axs[row*2, col].axis('off')
      
      # Plotting the attention image on the second and fourth rows.
      axs[row*2 + 1, col].imshow(attn_img, cmap='PuBuGn')
      axs[row*2 + 1, col].set_title(f'Attention {title} {idx+1}')
      axs[row*2 + 1, col].axis('off')
    else:
      axs.ravel()[idx].imshow(attn_img, cmap='PuBuGn')
      axs.ravel()[idx].set_title(f'Attention {title} {idx+1}')
      axs.ravel()[idx].axis('off')
      
    return axs
    
  #################################
    
  def save_plot(self, fig, name):
    # Adjusting the layout to avoid overlapping of labels and images.
    #fig.tight_layout()
    
    # Saving each figure as a separate file.
    fig.savefig(f'{self.out_folder}/{name}.png')
    if self.verbose:
      print(f'Info:: Figure saved in {self.out_folder}/{name}.png')
      
    # Closing the current figure to avoid overlapping of images in the next iteration.
    plt.close()
        
  #################################

  def plot_vs_vertical_level(self, ds):
    print("Info:: Start plotting as a function of the vertical level.")
    data = ds[grib_index[self.field]].isel(datetime = self.rf_time)
    vmin, vmax = data.min(), data.max()
    for k in range(data.ml.shape[0]):
      fig = plt.figure( figsize=(10,5), dpi=300)
      ax = plt.axes( projection=cartopy.crs.Robinson( central_longitude=0.))
      ax.add_feature( cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
      ax.set_global()
      level = ds['ml'].values[k]
      date = ds['datetime'][self.rf_time].astype('datetime64[m]')
      ax.set_title(f'{self.field} : level {level} - {date} ')
      data.isel(ml = k).plot.imshow(cmap=self.cmap, vmin=vmin, vmax=vmax)
      #    im = ax.imshow( np.flip(ds[grib_index[field]].values[rf_lvl, k], 0), cmap=cmap, vmin=vmin, vmax=vmax,
      im = ax.imshow( data.isel(ml = k), cmap=self.cmap, vmin=vmin, vmax=vmax,
                      transform=cartopy.crs.PlateCarree( central_longitude=180.))
      axins = inset_axes( ax, width="80%", height="5%", loc='lower center', borderpad=-2 )
      self.save_plot(fig, name = f"Combined_plot_VLEVEL_{self.tag}_{self.field}_level_{k}")

    #################################

  def plot_vs_time_step(self, ds):
    print("Info:: Start plotting as a function of the time step.")
    data = ds[grib_index[self.field]].isel(ml = self.rf_lvl)
    vmin, vmax = data.min(), data.max()
    
    for k in range(ds.datetime.shape[0]):
      fig = plt.figure( figsize=(10,5), dpi=300)
      ax = plt.axes( projection=cartopy.crs.Robinson( central_longitude=0.))
      ax.add_feature( cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
      ax.set_global()
      date = ds['datetime'].values[k].astype('datetime64[m]')
      ax.set_title(f'{self.field} : {date}')
      data.isel(datetime = k).plot.imshow(cmap=self.cmap, vmin=vmin, vmax=vmax)
      #    im = ax.imshow( np.flip(ds[grib_index[field]].values[rf_lvl, k], 0), cmap=cmap, vmin=vmin, vmax=vmax,
      im = ax.imshow( data.isel(datetime = k), cmap=self.cmap, vmin=vmin, vmax=vmax,
                      transform=cartopy.crs.PlateCarree( central_longitude=180.))
      axins = inset_axes( ax, width="80%", height="5%", loc='lower center', borderpad=-2 )
      self.save_plot(fig, name = f"Combined_plot_TIME_{self.tag}_{self.field}_time_{k:03d}")
      
  #################################

  def plot_vs_head_number(field, source, attn, label, mode = "accum", verbose = False):
    print("Info:: Start plotting as a function of the Head number.")
    for batch1 in range(attn.shape[0]):
      #    for batch2 in range(attn.shape[2]):
      # Create a gridspec to control the layout
      fig = plt.figure(figsize=(16, 20))
      gs = gridspec.GridSpec(6, 4)
      
      # Add the source image at the top, now twice as large
      source_ax = fig.add_subplot(gs[0:2, :])
      source_img = source[batch1, 0, 0, :, :]
      source_ax.imshow(source_img, cmap='PuBuGn')
      source_ax.set_title('Source Image', fontsize=16)
      source_ax.axis('off')
      
      # Add each of the attention images in a 4x4 grid beneath the source image
      for head in range(attn.shape[1]):
        ax = fig.add_subplot(gs[head // 4 + 2, head % 4])  # start from the third row
        att_img = attn[batch1, head, batch2, 0, 0, :, :]
        if soft_max:
          att_img = compute_softmax(att_img)
        ax.imshow(att_img, cmap='PuBuGn')
        ax.set_title(f'Attention Head {head+1}')
        ax.axis('off')

      # Save the figure
      plt.savefig(f'{out_folder}/{field}Combined_plot_HEADS_{label}_batch1_{batch1}_batch2_{batch2}.png')

      # Close the figure to free memory
      plt.close(fig)
      if verbose:
        print(f'Info:: Figure saved in {out_folder}/{field}Combined_plot_HEADS_{label}_batch1_{batch1}_batch2_{batch2}.png')
      
