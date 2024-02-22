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

from attention_utils import HandleAtmoRepAttention
from utils.utils import shape_from_str, grib_index

base_dir     = '/p/scratch/atmo-rep/results/'
#model_id     = 'ehsjzkbh' #big run
model_id     = '1zq3f5gf' #'8x5dnvc9' #'bkkrobb9' #'xsbasstk' #'q7yr6hrj'
out_folder   = "./attention_plots/"
tag          = model_id+"_test" #modify it if you want to add a particular description to your plots

#TO-DO: automatize this
num_batches_per_global = 14
rng = np.random.default_rng()
  
#reference coordinates:
#coordinates of the token to which the attention is referred to for each local patch. 
rf_dict = {"head" : 0, #--> the plots (unless when plottig as a function of the heads) refer to this head
           "layer": 0, #--> the plots refer to this layer
           
           "lvl" : 0,  #--> reference level
           "time": 0,  #--> reference time
           "lat" : 3,  #--> reference latitude
           "lon" : 6}  #--> reference longitude

##################################  
def plot_attention(field, filename, toksize = [12, 6, 12]):
  
  ar = HandleAtmoRepAttention(field, out_folder, tag, rf_dict)
  attn = ar.load_attention(filename)
  print("Info:: Start creating the dataset.")
  print(attn.info)

  #loop over lead times
  #TO-DO: remove num_batches_per_global
  for idx_nb in range(int(len(attn.keys()) / num_batches_per_global)): 
    batch = idx_nb*num_batches_per_global
    ar.set_tag(tag+f"_batch{batch:05d}")
    temp = ar.get_layer(attn, batch = batch, layer = rf_dict["layer"])
    datetime = temp["datetime"]
    ds_o = xr.Dataset( coords={ 'ml': temp["ml"],  
                                'datetime': np.unique(datetime),
                                'lat' : np.linspace( 0., 180., num=81, endpoint=True),
                                #'lat' : np.linspace( -90., 90., num=81, endpoint=True),
                                'lon' : np.linspace( 0., 360., num=160, endpoint=False) } )
    ds_o[grib_index[field]] = (['ml', 'datetime', 'lat', 'lon'], np.zeros( ( len(temp["ml"]), toksize[0], 81, 160)))

    
    #TO-DO: skipping the last batch since it's problematic - Fix it
    for b1 in range(batch, batch + datetime.shape[0]-1):
      lat, lon = ar.get_lat_lon(attn, batch = b1, layer = rf_dict["layer"])
      temp = ar.get_head(attn, batch = b1, layer = rf_dict["layer"], head = rf_dict["head"])
      temp = temp.reshape(*temp.shape[:3], toksize[1], toksize[2], *temp.shape[-3:-1], toksize[1], toksize[2])
      for b in range(datetime.shape[0]):
        ds_o[grib_index[field]].loc[ dict(datetime = datetime[b], 
                                          lat=lat[b],
                                          lon=lon[b]) ] = temp[b, rf_dict["lvl"], rf_dict["time"], rf_dict["lat"], rf_dict["lon"]]
        
    print(ds_o)
    print("Info:: Start plotting.")
    
    ############# PLOT x LEVEL #############
    ar.plot_vs_vertical_level(ds_o)
    
    ############ PLOT x TIME STEP ##########
    ar.plot_vs_time_step(ds_o)
  
  return 0

def compare_to_source(field, filename, toksize = [12, 6, 12], num_samples = 2):

  # get source if available
  fname_source = glob.glob(base_dir +'/id'+ model_id +'/*source*.zarr')
  if fname_source != []:
    fname_source = fname_source[0]
    store = zarr.ZipStore(fname_source, mode='r')
    root_source = zarr.open_group(store)[field]
    rnd_samples = rng.permutation(len(root_source.keys()))[:num_samples]
    source = [root_source[f'/{field}/sample={rnd:05d}/'].data[:, ::3] for rnd in rnd_samples]
    print (source[0].shape)
  else:
    rnd_samples = rng.permutation(num_batches_per_global**2)[:num_samples] #get random samples from first day only: TO-DO - improve this
    source = None

  #attention
  ar = HandleAtmoRepAttention(field, out_folder, tag, rf_dict)
  ds_att = ar.load_attention(filename)
  print("Info:: Start creating the dataset.")
  print(ds_att.info)

  #default shape
  attn = []
  for idx_nb in rnd_samples:
    print("Info:: processing random sample number: ", idx_nb)
    ext_loop = int(np.floor(idx_nb / num_batches_per_global))
    int_loop = int(idx_nb % num_batches_per_global)
    temp = ar.get_layer(ds_att, batch = ext_loop, layer = rf_dict["layer"])
    nheads = len(temp["heads"].keys())
    nbatch = temp["datetime"].shape[0]
    nlevel = temp.ml.shape[0]
    attn_data = np.zeros( ( nheads, nlevel, *toksize))
    
    #retrieve all heads for the given sample
    for head in range( nheads):
      temp = ar.get_head(ds_att, batch = ext_loop, layer = rf_dict["layer"], head = head)[int_loop] #get the specific local sample
      temp = temp.reshape(*temp.shape[:2], toksize[1], toksize[2], *temp.shape[-3:-1], toksize[1], toksize[2])
      temp = temp[rf_dict["lvl"], rf_dict["time"], rf_dict["lat"], rf_dict["lon"]]
      attn_data[head] = temp
    attn.append(attn_data)

  print("Info:: Start plotting.")
  
  ############# PLOT x LEVEL #############
  ar.compare_levels(attn, source, rnd_samples)

  ############ PLOT x TIME STEP ##########
  ar.compare_time(attn, source, rnd_samples)

  ############ PLOT x HEAD NUMBER ########
  ar.compare_heads(attn, source, rnd_samples)

  
##########################################

def main():
  bname = base_dir + '/id' + model_id + '/'

  with open( bname + '/model_id'+ model_id + '.json') as json_file:
    config = json.load(json_file)
  num_heads = config["encoder_num_heads"]

  # output
  if not os.path.exists(out_folder):
    os.makedirs( out_folder)
  print( 'Info:: Output for id = \'{}\' written to: {}'.format( model_id, out_folder))

  #loop over fields
  for ifield, field_info in enumerate(config['fields']) :
    field = field_info[0]
    toksize = field_info[3]
    
    print( 'Info:: Processing {}.'.format( field ) )

    #attention file
    filename = glob.glob(base_dir +'/id'+ model_id +'/*attention*.zarr')[0]
        
    #plot 2D maps
    plot_attention(field, filename, toksize)

    #plot single examples vs source
    compare_to_source(field, filename, toksize)
    
if __name__ == '__main__':
  main()
