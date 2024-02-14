import numpy as np
import glob
import json
import datetime
import os
import code
import xarray as xr
import pandas
import zarr

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

from analysis.utils.utils import shape_from_str, grib_index

def sort_key(filename):
  # Split the filename into parts based on '/'. The last part is something like '2.pth'. We split this based on '.'
  last_part = filename.split('/')[-1].split('.')
  # The first part of this is the integer we want
  return int(''.join(filter(str.isdigit, last_part[0])))

#################################

def compute_softmax(x):
  softmax = torch.nn.Softmax(dim=-1)
  return softmax(torch.from_numpy(x)).detach().cpu().numpy()

#################################

def inspect_attention(attn):
  print(type(attn))
  print(attn.shape)

#################################

def load_attention(filename, field, stack = False, mean = False):
  attn = []
  store = zarr.ZipStore(filename, mode='r')
  root = zarr.open_group(store)[field]
  for name, sample in root.groups():
      attn_temp = sample.data[:]
      if soft_max:
        attn_temp =  (attn_temp - attn_temp.mean()) / attn_temp.std()
      if stack:
        attn_temp =  np.stack(attn_temp)
      if mean:
        print(type(attn_temp), attn_temp.shape)
        attn_temp = attn_temp.mean(dim = [-3:])
        
      attn.append(attn_temp)

  attn = np.array(attn, dtype=np.float32)
  inspect_attention(attn)
  print("Info:: Attention maps loaded successfully.")
  return  attn


#################################

def add_plot(axs, source_img, attn_img, idx, nrows, ncols, title):

  if soft_max:
    attn_img = compute_softmax(attn_img)

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

  return axs

#################################

def save_plot(fig, name, verbose = False):
  # Adjusting the layout to avoid overlapping of labels and images.
  fig.tight_layout()

  # Saving each figure as a separate file.
  fig.savefig(f'{out_folder}/{name}.png')
  if verbose:
    print(f'Info:: Figure saved in {out_folder}/{name}.png')

  # Closing the current figure to avoid overlapping of images in the next iteration.
  plt.close()

#################################

def plot_vs_batch_number(field, source, attn, label, verbose = False):
  print("Info:: Start plotting as a function of the Batch number.")
  for batch1 in range(attn.shape[0]): #14
    for head in range(attn.shape[1]): #16
      fig, axs = plt.subplots(4, 7, figsize=(20, 10)) # Creating a grid of 4 rows and 7 columns
      # Loop over the second dimension.
      #for batch2 in range(attn.shape[2]): #14
      # Extracting the tensor and the attention for the given indices.
      source_img = source[batch1, batch2, 0, 0, :, :]
      att_img = attn[batch1, head, batch2, 0, 0, :, :]
      add_plot(axs, source_img, att_img, idx = head, nrows = 4, ncols = 7, title = "Batch")
      save_plot(fig, name = f"Combined_plot_BATCH_{label}_{field}_batch1_{batch1}_head_{head}", verbose = verbose)

#################################

def plot_vs_vertical_level(field, source, attn, label, verbose = False):
  print("Info:: Start plotting as a function of the vertical level.")
  for batch1 in range(attn.shape[0]): #14
    for batch2 in range(attn.shape[2]): #14
      fig, axs = plt.subplots(2, 5, figsize=(20, 10)) # Creating a grid of 4 rows and 7 columns

      # Loop over the vertical level (5)
      for lvl in range(attn.shape[3]): #5
        # Extracting the tensor and the attention for the given indices.
        source_img = source[batch1, batch2, lvl, 0, :, :]
        att_img = attn[batch1, 0, batch2, lvl, 0, :, :]

        add_plot(axs, source_img, att_img, idx = lvl, nrows = 2, ncols = 5, title = 'Level')
      save_plot(fig, name = f"Combined_plot_VLEVEL_{label}_{field}_batch1_{batch1}_batch2_{batch2}", verbose = verbose)

#################################

def plot_vs_time_step(field, source, attn, label, verbose = False):
  print("Info:: Start plotting as a function of the time step.")
  for batch1 in range(attn.shape[0]):
    for batch2 in range(attn.shape[2]):
      fig, axs = plt.subplots(4, 6, figsize=(20, 10)) # Creating a grid of 4 rows and 7 columns
      
      # Loop over the second dimension.
      for time in range(attn.shape[4]):
      # Extracting the tensor and the attention for the given indices.
        source_img = source[batch1, batch2, 0, time, :, :]
        att_img = attn[batch1, 0, batch2, 0, time, :, :]
        add_plot(axs, source_img, att_img, idx = time, nrows = 4, ncols = 6, title = 'Time')

      save_plot(fig, name = f"Combined_plot_TIME_{label}_{field}_batch1_{batch1}_batch2_{batch2}", verbose = verbose)
      
#################################

def plot_vs_head_number(field, source, attn, label, verbose = False):
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
      
