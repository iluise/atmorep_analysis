####################################################################################################
#
#  Copyright (C) 2022, 2023
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
# 
#  description :
#
#  license     :
#
####################################################################################################

import numpy as np
import zarr
import xarray as xr
import os 

import cartopy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.linewidth'] = 0.1
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plot_target   = False
plot_ensemble = False #False
#model_id = 'qsyjakm5' #'wc1nz35x' #'jyqf96nx'
#field = 'temperature' #'temperature'
#model_id = 'i96h8thv' 
# model_id = 'o8ryeqwt'
#model_id = 'c0xavsnq'
#model_id = 'w4o10vfk'
# model_id= 'yljwuxbf'
# model_id='qxsjh2g3'
#model_id='32y7kl0y' #preprint - new code all levels
#model_id='whr3w7d9' #preprint model - main
#model_id='87syt863' #preprint model - all levels - main
#model_id='i96vl336' #continue training
#model_id='4qo7s9xd' #CRPS
model_id='fbaizg2v' # MSE+stats
#model_id='z5vc2b9j' #weightedMSE + stats
#model_id='rd4c6tbu' #12 dates
field ='velocity_u'

#model_id = '7ks4klpl'
#model_id='v23oymqn'
#model_id='2vnb2awm'
#model_id='d2wjesm4' #pre-print model - main
#model_id='lzdvxquf' #continue runs
#model_id = '7ij1hjjb'
#model_id='zbyrwt3i' #CRPS loss
#model_id='nlbxpfqq' #MSE ensemble + stats
#model_id='ifddmor9' #weighted MSE + stats
# model_id='2vrhlmj5' #stats
#field='temperature'

#model_id='7gacxauw'
#model_id='240e6d7a'
#model_id='bh7xb1t5'
# model_id='9ij898wm' #pre-print model - main
# field = 'specific_humidity'

ml_level = 114


########################################

def get_weights(weight = True):
  if(weight):
    lat_sum = np.sum(np.cos(np.arange( 90.* np.pi/180., -90. * np.pi/180., -np.pi/721.)))
   
    theta_weight = np.array([721 * np.cos(w)/lat_sum for w in np.arange( 90. * np.pi/180. , -90. * np.pi/180., -np.pi/721.)], dtype = np.float32)
    theta_weight = np.repeat(theta_weight[:, np.newaxis], 1440, axis = 1)
  else:
    theta_weight = 1

  return theta_weight

########################################

def calc_rmse(targets, preds, weight = True):

  theta_weight = get_weights(weight)
  # plt.imshow(theta_weight * ((preds - targets )**2), cmap = 'RdBu_r')
  # plt.savefig("mse.png")
  return np.sqrt(( theta_weight * ((preds - targets )**2)).mean()).values

def calc_ssr(targets, ens_mean, ens_std):
  breakpoint()
  spread = np.zeros( [targets.shape[0], targets.shape[1]] )
  for ilevel in range( targets.shape[0]) :
    for i in range(targets.shape[1]) :
      spread[ilevel, i] = np.sqrt(np.sum( theta_weight * ((ens_std[ilevel,i])**2)) / (targets.shape[2] * targets.shape[3]))
    return spread/calc_rmse(targets, ens_mean, weight = weight)

########################################

def get_data(target = False, ensemble = False, ens_member = 0):

  if target and ensemble:
    ValueError("choose target or ensemble view.")

  if target:
    store = zarr.ZipStore( f'/gpfs/scratch/ehpc03/results/id{model_id}/results_id{model_id}_epoch00000_target.zarr')
  elif ensemble:
    store = zarr.ZipStore( f'/gpfs/scratch/ehpc03/results/id{model_id}/results_id{model_id}_epoch00000_ens.zarr')
  else:
    store = zarr.ZipStore( f'/gpfs/scratch/ehpc03/results/id{model_id}/results_id{model_id}_epoch00000_pred.zarr')
  ds = zarr.group( store=store)
  # create empty canvas where local patches can be filled in
  # use i=0 as template; structure is regular
  i = 0
  ds_o = xr.Dataset( coords={ 'ml' : ds[ f'{field}/sample={i:05d}/ml' ][:],
                              'datetime': ds[ f'{field}/sample={i:05d}/datetime' ][:], 
                              'lat' : np.linspace( -90., 90., num=180*4+1, endpoint=True), 
                              'lon' : np.linspace( 0., 360., num=360*4, endpoint=False) } )
  nlevels = ds[ f'{field}/sample={i:05d}/ml' ].shape[0]
  ds_o['field'] = (['ml', 'datetime', 'lat', 'lon'], np.zeros( ( nlevels, 6, 721, 1440)))
  #breakpoint()
  # fill in local patches
  for i_str in ds[ f'{field}']:
    if np.any(ds[ f'{field}/{i_str}/datetime' ][:]  != ds_o['field'].datetime):
      break
    if ensemble:
      ds_o['field'].loc[ dict( datetime=ds[ f'{field}/{i_str}/datetime' ][:],
          lat=ds[ f'{field}/{i_str}/lat' ][:],
          lon=ds[ f'{field}/{i_str}/lon' ][:]) ] = ds[ f'{field}/{i_str}/data'][:][ens_member, :]
    else:
      ds_o['field'].loc[ dict( datetime=ds[ f'{field}/{i_str}/datetime' ][:],
          lat=ds[ f'{field}/{i_str}/lat' ][:],
          lon=ds[ f'{field}/{i_str}/lon' ][:]) ] = ds[ f'{field}/{i_str}/data'][:] #[ens_member, :]
  return ds_o

ds_target = get_data(target = True)
vmin, vmax = ds_target['field'].values[0].min(), ds_target['field'].values[0].max()

if plot_target:
  for k in range(ds_o['field'].datetime.shape[0]): #range( 6) :
    fig, ax = plt.subplots(1,1)
    date = ds_target['datetime'].values[k].astype('datetime64[m]')
    ax.set_title(f'{field} : {date}')
    ds_target['field'].sel(ml=ml_level).isel(datetime = k).plot.imshow(cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig( f'{model_id}/example_{model_id}_{k:03d}_target.png')
    plt.savefig( f'{model_id}/example_{model_id}_{k:03d}_target.pdf')
    plt.close()

#TODO: make it nicer
ens_num = 16 if plot_ensemble else 1

for ens_member in range(ens_num):
  ds_o = get_data(ensemble = plot_ensemble, ens_member = ens_member)

  if not os.path.exists(model_id):
      os.makedirs(model_id)

  # plot and save the three time steps that form a token
  cmap = 'RdBu_r'
 
  print(ds_o['field'].datetime.shape)
  for k in range(ds_o['field'].datetime.shape[0]): #range( 6) :
    #fig = plt.figure( figsize=(10,5), dpi=300)
    #ax = plt.axes( projection=cartopy.crs.Robinson( central_longitude=0.))
    #ax.add_feature( cartopy.feature.COASTLINE, linewidth=0.5, edgecolor='k', alpha=0.5)
    #ax.set_global()
    fig, ax = plt.subplots(1,1)
    date = ds_o['datetime'].values[k].astype('datetime64[m]')
    ax.set_title(f'{field} : {date}')
    ds_o['field'].sel(ml=ml_level).isel(datetime = k).plot.imshow(cmap=cmap, vmin=vmin, vmax=vmax)
    im = ax.imshow( np.flip(ds_o['field'].values[0,k], 0), cmap=cmap, vmin=vmin, vmax=vmax)
                    #transform=cartopy.crs.PlateCarree( central_longitude=180.))
    #axins = inset_axes( ax, width="3%", height="80%", loc='lower right', borderpad=-2 )
    #fig.colorbar( im, cax=axins) #, orientation="horizontal")
    if plot_ensemble:
      plt.savefig( f'{model_id}/example_{model_id}_{k:03d}_ens{ens_member}.png')
      plt.savefig( f'{model_id}/example_{model_id}_{k:03d}_ens{ens_member}.pdf')
    else:
      plt.savefig( f'{model_id}/example_{model_id}_{k:03d}.png')
      plt.savefig( f'{model_id}/example_{model_id}_{k:03d}.pdf')
    print("RMSE:", calc_rmse(ds_o['field'].sel(ml=ml_level).isel(datetime = k), ds_target['field'].sel(ml=ml_level).isel(datetime = k)))
    plt.close()