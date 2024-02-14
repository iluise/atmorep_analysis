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

#for maps
import cartopy
import cartopy.crs as ccrs  #https://scitools.org.uk/cartopy/docs/latest/installing.html
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

from analysis.attention.attention_utils import *
from analysis.utils.utils import shape_from_str, grib_index

base_dir     = './results/'
model_id     =  '8x5dnvc9' #'bkkrobb9' #'xsbasstk' #'q7yr6hrj'
out_folder   = "./attention_plots/"
epoch        = 0
num_samples  = 10
accumulate   = False
soft_max     = False
single_token = True
  
##################################
  
def plot_accumulated_attention(field, filename, source, idxs_nb):
  
  attn = load_attention(filename, mean = True) 
  print(attn.shape)
  for batch in range(attn.shape[0]): #local patch
      fig, axs = plt.subplots(4, 6, figsize=(20, 10)) # Creating a grid of 4 rows and 7 columns
      for time in range(attn.shape[3]): #time
        # Extracting the tensor and the attention for the given indices.
        source_img = source[batch1, 0, time, :, :]
        attn_img = attn[batch1, 0, time, :, :]
        add_plot(axs, source_img, attn_img, idx = time, nrows = 4, ncols = 6, title = 'Time')

        sm = "_SM" if soft_max else ""
      save_plot(fig, name = f"Combined_plot_{model_id}_{field}_batch1_{batch1}_batch2_{batch2}_accum"+("_SM" if soft_max else ""))
        
#################################

def plot_attention(field, filename, source, idxs_nb):

  ########## SINGLE TOKEN or AVERAGE ###############
  
  if single_token:
    lvl = 0 #4 #last vertical level (137)
    time = 0 #first timestep 
    
    print(f"Info:: Plotting Attention Maps referred to the Single Token: lvl {lvl}/{num_lvl}, time {time}/{num_tokens[0]}, lat {lat}/{num_tokens[1]}, lon = {lon}/{num_tokens[2]}")
    attn = []
    #attn_temp = np.fromfile(f, dtype = np.float32).reshape(shape_from_str(f))
    #inspect_attention(attn_temp)
    store = zarr.ZipStore(filename, mode='r')
    root = zarr.open_group(store)[field]
    attn_list = []
    #attn = np.zeros([1488, 16, 2, 12, 6, 12, 2, 12, 6, 12], dtype = np.float32)

    attn = load_attention(filename, field, stack = False, mean = False)[:, :, lvl,time,lat,lon,:,:,:,:]
    print(attn.shape)
  else: 
    print("Info:: Computing the Mean across tokens.")
    attn = load_attention(filename, stack = True, mean = True) 
    print(attn.shape)
    #attn = attn.reshape([attn.shape[0], num_heads, attn.shape[0], num_lvl, num_tokens[0], num_tokens[1], num_tokens[2]])
    #print(attn.shape)
    
  print("Info:: Start plotting.")

  label = model_id + ("_ST" if single_token else "" ) + ("_SM" if soft_max else "")

  ########### PLOT x BATCH ###############
  plot_vs_batch_number(field, source, attn, label)
  
  ############# PLOT x LEVEL #################
  plot_vs_vertical_level(field, source, attn, label)
  
  ############ PLOT x TIME STEP ################
  plot_vs_time_step(field, source, attn, label)
  
  ############## PLOT x HEADS ###################
  plot_vs_head_number(field, source, attn, label)

  return 0

  
###################################################################3            

def main():
  bname = base_dir + '/id' + model_id + '/'

  with open( bname + '/model_id'+ model_id + '.json') as json_file:
    config = json.load(json_file)
  
  print(config)
  num_heads = config["encoder_num_heads"]
  
  # output
  if not os.path.exists(out_folder):
    os.makedirs( out_folder)
  print( 'Info:: Output for id = \'{}\' written to: {}'.format( model_id, out_folder))
  more_tsteps = True
  tidx = 0

  cat = np.concatenate
  f32 = np.float32
  print("Info:: Accumulate  = ", accumulate)
  print("Info:: Soft Max    = ", soft_max  )
  if soft_max:
    print("Warning:: the soft max implementation is not exactly as in the code. A normalisation will be applied for better visualisation.")

  for ifield, field_info in enumerate(config['fields']) :

    field = field_info[0]
    print( 'Info:: Processing {}.'.format( field ) )

    # source
    fname_source = glob.glob(base_dir +'/id'+ model_id +'/*source*.zarr')[0]
    store = zarr.ZipStore(fname_source, mode='r')
    root_source = zarr.open_group(store)[field]
    source = []
    for name, sample in root_source.groups():
      source_temp = sample.data[:]
      source.append(source_temp)
    source = np.array(source)
    print( 'Info:: source : {}'.format( source.shape ) )

    rng = np.random.default_rng()
    idxs_nb = rng.permutation( source.shape[0] )[:num_samples]
    source = source[idxs_nb, :]
    print(source.shape)
    #attention file
    filename = glob.glob(base_dir +'/id'+ model_id +'/*attention*.zarr')[0]
    
    if accumulate:
      plot_accumulated_attention(field, filename, source, idxs_nb)
    else:
      plot_attention(field, filename, source, idxs_nb)
    
if __name__ == '__main__':
  main()