"""
Methods for creating plots.
"""

__authors__ = "Ilaria Luise, Michael Langguth"
__email__ = "ilaria.luise@cern.ch"
__date__ = "2023-12-20"
__update__ = "2025-01-13"

# for processing data
import os
from pathlib import Path
import logging
from typing import Union, List, Dict, Any
import numpy as np
import xarray as xr
import pandas as pd
from itertools import product
from xhistogram.xarray import histogram

# for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.linewidth'] = 0.1
import matplotlib.colors as mcolors

from mpl_toolkits.axes_grid1 import make_axes_locatable

#for maps
import cartopy
import cartopy.crs as ccrs  #https://scitools.org.uk/cartopy/docs/latest/installing.html
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
from analysis.utils.utils import get_units
# auxiliary variable for logger
module_name = os.path.basename(__file__).rstrip(".py")

str_or_path = Union[str, Path]  # type hint for string or Path objects

########################################
# Auxiliary functions
########################################

def CustomPalette():
  colors = [ (0.278, 0.380, 0.620), (0.867, 0.647, 0.365), (0.991, 0.949, 0.765)]
  cmap = mcolors.LinearSegmentedColormap.from_list('YlBu', colors, N=100)
  return cmap

def MathematicaPalette():
  colors = [ 
          (0.368417,	0.506779,	0.709798),
          (0.880722,	0.611041,	0.142051),
          (0.560181,	0.691569,	0.194885),
          (0.922526,	0.385626,	0.209179),
          (0.528488,	0.470624,	0.701351),
          (0.772079,	0.431554,	0.102387),
          (0.363898,	0.618501,	0.782349),
          (1.000000,	0.750000,	0),
          (0.647624,	0.378160,	0.614037),
          (0.571589,	0.586483,	0.),
          (0.915000,	0.332500,	0.2125),
          (0.400822,	0.522007,	0.85),
          (0.972829,	0.621644,	0.073362),
          (0.736783,	0.358000,	0.503027),
          (0.280264,	0.715000,	0.429209)
          ]
  cmap = mcolors.LinearSegmentedColormap.from_list('MathCol', colors, N=len(colors))
  return cmap

def get_cmap_norm(levels, cb_name: str = "PuOr_r", cb_range= (0., 1.)):
    """
    Get the colormap and norm-object for given levels and a given colorbar-name
    :param levels: level boundaries
    :param cb_name: name of colorbar 
    :return cmap: colormap-object
    :return norm: normalization object corresponding to colormap and levels
    """
    bounds = np.asarray(levels)
    nbounds = len(bounds)
    
    col_obj = plt.get_cmap(cb_name)
    col_obj = col_obj(np.linspace(*cb_range, nbounds)) 

    # create colormap and corresponding norm
    cmap = mpl.colors.ListedColormap(col_obj)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


########################################
   
def create_canvas(figsize = (8, 8), ncols = 1, nrows = 1):  
  fig, ax_temp = plt.subplots(figsize=(6, 6))
  gs = fig.add_gridspec(ncols, nrows)
  ax_temp.remove()
  ax = [fig.add_subplot(gs[c,r]) for c,r in product(range(ncols), range(nrows))]
  return fig, ax

########################################

def imshow(data, ax, title = '', vmin=None, vmax=None, colorbar = False, remove_ticks = False):
  im = ax.imshow(data, cmap=CustomPalette(), vmin=vmin, vmax=vmax)
  ax.set_title(title, color='dimgray')
  if remove_ticks:
    ax.set_xticks([]), ax.set_yticks([])
  return im

########################################

def plot(hlist, ax, linewidth = 1):
  for h in hlist:
    ax.plot(h[0], label = h[1], color = h[2], linewidth = 1.5)
  ax.legend(frameon=False)
  return ax

########################################

def plot_on_map(data, field, cmap = "RdBu", norm = None, zrange = [0.,0.1]):
    """
    plot on a world map
    """
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_global()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    #see: https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
    pos1 = ax.get_position() # get the original position 
    pos2 = [pos1.x0 - 0.04, pos1.y0,  pos1.width, pos1.height] 
    ax.set_position(pos2) # set a new position

    if(norm != None):
      im = plt.imshow(data, cmap=cmap, extent=[-180,180,-90,90], norm=norm) #or use plt.pcolor
    else:
      im = plt.imshow(data, cmap=cmap, extent=[-180,180,-90,90], vmin=zrange[0],  vmax=zrange[1]) #or use plt.pcolor
    cbar = plt.colorbar(im, shrink=0.7, cax=fig.add_axes([0.9, 0.23, 0.022, 0.5])) #, format='%.0e')
    cbar.set_label(get_units(field), y=-0.04, ha='right', rotation=0)
   # plt.savefig(name)
    return fig

########################################

def plot_on_map_custom_edges(data, edges, cmap = "RdBu", norm = None, zrange = [0.,0.1]):
    """
    plot on a world map
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_global()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
    # ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    im = plt.imshow(data, cmap=cmap, vmin=zrange[0],  vmax=zrange[1]) #or use plt.pcolor
    ax.set_ylim(min(edges[:2]), max(edges[:2]))
    ax.set_xlim(min(edges[2:]), max(edges[2:])) 
    plt.colorbar(im, orientation="horizontal", shrink=0.7)
   # plt.savefig(name)
    return fig

def plot_1D_wDiff(field, data1, data2, label1, label2):
  
    fig1, ax_temp = plt.subplots(figsize=(6, 6)) 
    gs = fig1.add_gridspec(ncols = 1, nrows = 2, height_ratios = [3,1])
    ax_temp.remove()
    axs = []
    axs.append(fig1.add_subplot(gs[0,0]))
    axs.append(fig1.add_subplot(gs[1,0]))
    # axs.append(fig1.add_subplot(gs[2,0]))
    data1_f = data1.flatten()
    data2_f = data2.flatten()
    xmin = min(np.minimum(data1_f, data2_f))
    xmax = max(np.maximum(data1_f, data2_f))
    
    plt1 = axs[0].hist( data1_f, bins=50, fill=False, label = label1, range = [xmin, xmax], color='royalblue', histtype = 'step')
    plt2 = axs[0].hist( data2_f, bins=50, fill=False, label = label2,  range= [xmin, xmax], color='red', histtype = 'step')

    axs[0].set_xlim([xmin, xmax])
    axs[0].legend(frameon=False)
    
    #ratio and diff plots 
    width = (plt1[1][0] - plt1[1][-1])/len(plt1[1])
    diff = (plt1[0]-plt2[0])-width
    xvalues = plt1[1][:-1]-width
    axs[1].axhline(y=0., color='darkgray', linestyle='-', linewidth=0.5)
    axs[1].bar(xvalues, height=diff,
             width = width, align = 'edge')
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(2,2))
    axs[1].set_ylabel(label1+"-"+label2)
    axs[1].set_xlim([xmin, xmax])

    # ratio = np.divide(plt1[0], plt2[0])-width
    # axs[2].axhline(y=1., color='darkgray', linestyle='-', linewidth=0.5)
    # axs[2].bar(plt1[1][:-1], height=ratio, color = 'grey',alpha = 0.5, 
    #          width= width , align = 'edge')
    # axs[2].set_xlabel(field)
    # axs[2].set_ylabel(label1+"/"+label2)
    # axs[2].set_ylim([0, 2])
    # axs[2].set_xlim([xmin, xmax])
    plt.tight_layout()
    fig1.align_ylabels()
    return fig1

########################################

# auxiliary function for colormap
def get_colormap_temp(levels=None):
    """
    Get a nice colormap for plotting topographic height
    :param levels: level boundaries
    :return cmap: colormap-object
    :return norm: normalization object corresponding to colormap and levels
    """
    bounds = np.asarray(levels)

    nbounds = len(bounds)
    col_obj = mpl.cm.seismic_r(np.linspace(0.5, 1., nbounds))

    # create colormap and corresponding norm
    cmap = mpl.colors.ListedColormap(col_obj, name="temp" + "_map")
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm, bounds


# for making plot nice
def decorate_plot(ax_plot, plot_xlabel=True, plot_ylabel=True):
    fs = 16
    # if "login" in host:
    # add nice coast- and borderlines
    ax_plot.coastlines(linewidth=0.75)
    ax_plot.coastlines(linewidth=0.75)
    ax_plot.add_feature(cartopy.feature.BORDERS)

    # adjust extent and ticks as well as axis-label
    ax_plot.set_xticks(np.arange(0., 360. + 0.1, 2.))  # ,crs=projection_crs)
    ax_plot.set_yticks(np.arange(-90., 90. + 0.1, 2.))  # ,crs=projection_crs)

    ax_plot.set_extent([4., 17, 46., 56.])    # , crs=prj_crs)
    ax_plot.minorticks_on()
    ax_plot.tick_params(axis="both", which="both", direction="out", labelsize=fs)

    # some labels
    if plot_xlabel:
        ax_plot.set_xlabel("Longitude [°E]", fontsize=fs)
    if plot_ylabel:
        ax_plot.set_ylabel("Latitude[°N]", fontsize=fs)

    return ax_plot


# for creating plot
def create_mapplot(data1, data2, plt_fname, opt_plot={}):
    # get coordinate data
    try:
        time, lat, lon = data1["time"].values, data1["lat"].values, data1["lon"].values
        time_stamp = (pd.to_datetime(time)).strftime("%Y-%m-%d %H:00 UTC")
    except Exception as err:
        print("Failed to retrieve coordinates from data1")
        raise err
    # construct array for edges of grid points
    dy, dx = np.round((lat[1] - lat[0]), 2), np.round((lon[1] - lon[0]), 2)
    lat_e, lon_e = np.arange(lat[0]-dy/2, lat[-1]+dy, dy), np.arange(lon[0]-dx/2, lon[-1]+dx, dx)

    title1, title2 = opt_plot.get("title1", "input T2m"), opt_plot.get("title2", "target T2m")
    title1, title2 = "{0}, {1}".format(title1, time_stamp), "{0}, {1}".format(title2, time_stamp)
    levels = opt_plot.get("levels", np.arange(-5., 25., 1.))

    # get colormap
    cmap_temp, norm_temp, lvl = get_colormap_temp(levels)
    # create plot objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True,
                                   subplot_kw={"projection": ccrs.PlateCarree()})

    # perform plotting
    _ = ax1.pcolormesh(lon_e, lat_e, np.squeeze(data1.values), cmap=cmap_temp, norm=norm_temp)
    temp2 = ax2.pcolormesh(lon_e, lat_e, np.squeeze(data2.values), cmap=cmap_temp, norm=norm_temp)

    ax1, ax2 = decorate_plot(ax1), decorate_plot(ax2, plot_ylabel=False)

    ax1.set_title(title1, size=14)
    ax2.set_title(title2, size=14)

    # add colorbar
    cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(temp2, cax=cax, orientation="vertical", ticks=lvl[1::2])
    cbar.ax.tick_params(labelsize=12)

    # save plot and close figure
    plt_fname = plt_fname + ".png" if not plt_fname.endswith(".png") else plt_fname
    print(f"Save plot in file '{plt_fname}'")
    fig.savefig(plt_fname, bbox_inches="tight")
    plt.close(fig)


def save_ims( dir_out, field, data, name, min_val = 1., max_val = -1.) :
  
  if not os.path.exists(dir_out ):
    os.makedirs( dir_out)

  cmap = mpl.colormaps.get_cmap('PuBuGn')

  if 1. == min_val : 
    min_val = data.min()
  if -1. == max_val : 
    max_val = data.max()
  print( 'min / max : {} / {}'.format( min_val, max_val) )
  print(data.min(), data.max())
  bname = dir_out + '/fig_{}_{}.{}'
  fname = bname.format( field, name, 'png' )
  plt.imsave( fname, data, vmin=min_val, vmax=max_val, cmap = cmap )
  fname = bname.format( field, name, 'pdf' )
  plt.imsave( fname, data, vmin=min_val, vmax=max_val, cmap = cmap )
  plt.close()
    # print( 'Finished saving figures for step={}, tidx = {}.'.format( epoch, tidx) )

##############################################
# Plot routines used in the downscaling evaluation pipeline
##############################################

def mapplot_comparison_ens(data_ref: xr.DataArray, data_fcst: xr.DataArray, plt_fname: str_or_path, lshow: bool=True,
                           ens_name: str = "ens", **plt_kwargs):
    """
    Plot geographical reference/ground truth data and data from n ensemble members in a column
    :param data_ref: ground truth data
    :param data_fcst: forecast data with ensemble dimension (see ens_name-parameter)
    :param plt_fname: path to png-file where plot will be saved
    :param lshow: flag to show plot (in a Jupyter Notebook)
    :param ens_name: ensemble dimension name of data_fcst
    :param plt_kwargs: other plot parameters
                       valid parameter keys are:
                       - nens: number of ensemble members to plot (default: 3)
                        - figsize: figure size (default: (9, 6*nens))
                        - aspect_ratio: aspect ratio of the map (default: 2./3.)
                        - projection: cartopy projection-object used for the map (default: ccrs.PlateCarree())
                        - transform: cartopy transform-object used for the data (default: copied from projection-parameter)
                        - cmap_name: name of the colormap used for the plot (default: "coolwarm")
                        - levels: levels for the colormap (default: np.arange(-30, 31, 2))
                        - cmap_range: range for the colormap (default: (0., 1.))
                        - extent: geographical extent of the map [west, east, south, north] in degree (default: [-25, 40, 20, 75])
                        - unit: unit of the data (default: "kg m**-2")
                        - titles: list of titles for the two plots (default: None)
                        - sup_title: super title for the plot (default: None)
                        - fs: basic font size used in plot labels (default: 14)
    """
    nens = plt_kwargs.pop("nens", 3)
    figsize = plt_kwargs.pop("figsize", (9, 6*nens))
    aspect = plt_kwargs.pop("aspect_ratio", 2./3.)
    proj = plt_kwargs.pop("projection", ccrs.PlateCarree())
    transform = plt_kwargs.pop("transform", proj)
    cmap_name = plt_kwargs.pop("cmap_name", "coolwarm")
    levels = plt_kwargs.pop("levels", np.arange(-30, 31, 2))
    cb_range = plt_kwargs.pop("cmap_range", (0., 1.))
    extent = plt_kwargs.pop("extent", [-25, 40, 20, 75])
    unit = plt_kwargs.pop("unit", "kg m**-2")
    titles = plt_kwargs.pop("titles", None)
    suptitle = plt_kwargs.pop("sup_title", None)
    fs = plt_kwargs.pop("fs", 14)

    # Create a figure and an axes with a map projection
    fig, axes = plt.subplots(nens + 1, 1, subplot_kw={'projection': proj}, figsize=figsize)
    
    cmap, norm = get_cmap_norm(levels, cmap_name, cb_range= cb_range)

    data_ens = [data_fcst.isel({ens_name: iens}) for iens in range(nens)]
    
    for idx, (data, ax) in enumerate(zip([data_ref]  + data_ens, axes)):
        # Add map features for context
        ax.coastlines()
        # Add gridlines and customize labels
        gl = ax.gridlines(draw_labels=True)
        #gl.left_labels = True 
        gl.right_labels = False
        gl.top_labels = False

        data = data_ref if idx == 0 else data_fcst.isel({"ens": idx - 1})
        
        # Plot the data using contourf
        contour = ax.contourf(data["lon"], data["lat"], data, transform=transform,
                              cmap=cmap, norm=norm, levels=levels)

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_aspect(aspect)
        # control axis labels and the fontsizes
        ax.set_xlabel("Longitude", fontsize=fs)  
        ax.set_ylabel("Latitude", fontsize=fs)
        ax.tick_params(labelsize=fs-2)

        # Set title if provided
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=fs)
    
    # Add a shared colorbar for all plots
    cbar_ax = fig.add_axes([0.05, 0.01, 0.9, 0.015])  # [left, bottom, width, height]
    cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal', pad=0.05, ticks=levels, shrink=0.8)
    cbar.set_label(unit)
    
    # Adjust spacing between plots
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.2)
    if suptitle:
        plt.suptitle(suptitle, fontsize=fs+2)

    if lshow:
        plt.show()

    # save plot and close figure
    plt_fname = Path(plt_fname)
    plt_fname = plt_fname + ".png" if not plt_fname.suffix == ".png" else plt_fname
    print(f"Save plot in file '{plt_fname}'")
    fig.savefig(plt_fname, bbox_inches="tight")
    plt.close(fig)

def mapplot_comparison_det(data1: xr.DataArray, data2: xr.DataArray, plt_fname: str_or_path, lshow: bool=False, **plt_kwargs):
    """
    Plot two geographical data arrays on a map next to each other for comparison
    :param data1: first data array to plot
    :param data2 first data array to plot
    :param plt_fname: path to png-file where plot will be saved
    :param lshow: flag to show plot (set to True in a Jupyter Notebook)
    :param plt_kwargs: other plot parameters
                       valid parameter keys are:
                        - figsize: figure size (default: (12, 6))
                        - projection: cartopy projection-object used for the map (default: ccrs.PlateCarree())
                        - transform: cartopy transform-object used for the data (default: copied from projection-parameter)
                        - cmap_name: name of the colormap used for the plot (default: "coolwarm")
                        - levels: levels for the colormap (default: np.arange(-30, 31, 2))
                        - cmap_range: range for the colormap (default: (0., 1.))
                        - extent: geographical extent of the map [west, east, south, north] in degree (default: [-25, 40, 20, 75])
                        - unit: unit of the data (default: "kg m**-2")
                        - titles: list of titles for the two plots (default: None)
                        - fs: basic font size used in plot labels (default: 14)

    """
    figsize = plt_kwargs.pop("figsize", (12, 6))
    proj = plt_kwargs.pop("projection", ccrs.PlateCarree())
    transform = plt_kwargs.pop("transform", proj)
    cmap_name = plt_kwargs.pop("cmap_name", "coolwarm")
    levels = plt_kwargs.pop("levels", np.arange(-30, 31, 2))
    cb_range = plt_kwargs.pop("cmap_range", (0., 1.))
    extent = plt_kwargs.pop("extent", [-25, 40, 20, 75])
    unit = plt_kwargs.pop("unit", "kg m**-2")
    titles = plt_kwargs.pop("titles", None)
    fs = plt_kwargs.pop("fs", 14)

    # Create a figure and an axes with a map projection
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': proj}, figsize=figsize)
    
    cmap, norm = get_cmap_norm(levels, cmap_name, cb_range= cb_range)
    
    for idx, (data, ax) in enumerate(zip([data1, data2], axes)):
        # Add map features for context
        ax.coastlines()
        # Add gridlines and customize labels
        gl = ax.gridlines(draw_labels=True)
        gl.right_labels = idx == 1  # Only show right labels for the second plot
        gl.left_labels = idx == 0   # Only show left labels for the first plot

        # Plot the data using contourf
        contour = ax.contourf(data["lon"], data["lat"], data, transform=transform,
                              cmap=cmap, norm=norm, levels=levels)

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        # control axis labels and the fontsizes
        ax.set_xlabel("Longitude", fontsize=fs)  
        ax.set_ylabel("Latitude", fontsize=fs)
        ax.tick_params(labelsize=fs-2)
        
        # Set title if provided
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=fs)
    
    # Add a shared colorbar for both plots
    cbar = fig.colorbar(contour, ax=axes, orientation='horizontal', pad=0.05, ticks=levels, shrink=0.8)
    cbar.set_label(unit)
    # Add a colorbar
    #cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05, ticks=levels)
    #cbar.set_label(unit)
    
    if lshow:
        plt.show()

    # save plot and close figure
    plt_fname = Path(plt_fname)
    plt_fname = plt_fname + ".png" if not plt_fname.suffix == ".png" else plt_fname
    print(f"Save plot in file '{plt_fname}'")
    fig.savefig(plt_fname, bbox_inches="tight")
    plt.close(fig)

def plot_histogram(data1: xr.DataArray, data2: xr.DataArray, plt_fname: str_or_path, ens_dim: str = "ens" ,lshow: bool =False, **plt_kwargs):
    """
    Plot histogram of two data arrays next to each other.
    :param data1: first data array for which histogram shall be created
    :param data2 first data array for which histogram shall be created
    :param plt_fname: path to png-file where plot will be saved
    :param ens_name: ensemble dimension name of data_fcst
    :param lshow: flag to show histogram plot (set to True in a Jupyter Notebook)
    :param plt_kwargs: other histogram parameters
                        valid parameter keys are:
                        - bins: bins for histogram
                        - legend_labels: labels for the two histograms
                        - figsize: figure size (default: (9, 6))
                        - log_scale: flag for log-scale on y-axis (default: True)
                        - bar_colors: colors for the bars (default: ["blue", "green"])
                        - xlabel: x-axis label (default: "Precipitation Bins")
                        - plt_title: title of the plot (default: "Histogram of Hourly Precipitation")
                        - fs: basic font size used in plot labels (default: 14)
                        - bin_width: width of the bars plotted in the histogram (default: 0.8)
    """
    # get plot parameters
    bins_hist = plt_kwargs.pop("bins")
    legend_labels = plt_kwargs.pop("legend_labels")
    figsize = plt_kwargs.pop("figsize", (9, 6))
    yscale_log = plt_kwargs.pop("log_scale", True)
    bar_cols = plt_kwargs.pop("bar_colors", ["blue", "green"])
    xlabel = plt_kwargs.pop("xlabel", "Precipitation Bins")
    plt_title = plt_kwargs.pop("plt_title", "Histogram of Hourly Precipitation")
    fs = plt_kwargs.pop("fs", 14)
    bin_width = plt_kwargs.pop("bin_width", .8)

    # compute histogram data with awareness of ensemble-dimension
    # setting block size to NaN for unchunked arrays ensures that blocksize does not become zero in xhistogram 
    da1_hist = histogram(data1, bins=[bins_hist], dim=[dim for dim in data1.dims if dim != ens_dim], 
                         block_size=None if data1.chunks is None else "auto")
    da2_hist = histogram(data2, bins=[bins_hist], dim=[dim for dim in data1.dims if dim != ens_dim], 
                         block_size=None if data2.chunks is None else "auto")

    # average over ensemble-dim if present
    if ens_dim in da1_hist.dims: da1_hist = da1_hist.mean(ens_dim)
    if ens_dim in da1_hist.dims: da2_hist = da2_hist.mean(ens_dim)
    
    # Extract data and coordinates
    bin_edges = da1_hist[list(da1_hist.coords)[0]].values
    
    # Plot histogram
    fig, (ax) = plt.subplots(1, 1, figsize=figsize)
    
    offset = bin_width/4.
    tick_pos = np.arange(len(bin_edges))
    xlabels = [f"[{lvl}, {bins_hist[i+1]})" for i, lvl in enumerate(bins_hist[:-1])]
    
    # Plot first histogram 
    hist1 = ax.bar(tick_pos - offset, da1_hist.values, width=bin_width / 2, label=legend_labels[0], color=bar_cols[0])
    plt.xticks(tick_pos, xlabels)
    
    # Plot second histogram 
    hist2 = ax.bar(tick_pos + offset, da2_hist.values, width=bin_width / 2, label=legend_labels[1], color=bar_cols[1])
    
    # Set y-axis to log scale
    plt.legend()
    if yscale_log:
        plt.yscale('log')
    
    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel('Frequency', fontsize=fs)
    ax.set_title(plt_title, fontsize=fs)
    ax.tick_params(axis='both', labelsize=fs-2)
    
    # Display the plot
    if lshow:
        plt.show()

    # save plot and close figure
    plt_fname = Path(plt_fname)
    plt_fname = plt_fname + ".png" if not plt_fname.suffix == ".png" else plt_fname
    print(f"Save plot in file '{plt_fname}'")
    fig.savefig(plt_fname, bbox_inches="tight")
    plt.close(fig)

def plot_rank_histogram(rank_norm: xr.DataArray, plt_fname: str_or_path, lshow: bool = False, **plt_kwargs):
    """
    Plot normalized rank histogram.
    :param rank_norm: normalized rank histogram data (from rank_histogram function of Scores class)
    :param plt_fname: path to png-file where plot will be saved
    :param lshow: flag to show plot (set to True in a Jupyter Notebook)
    :param plt_kwargs: other plot parameters
                        valid parameter keys are:
                        - figsize: figure size (default: (9, 6))
                        - line_color: color of the line plot (default: "blue")
                        - linestyle: style of the line plot (default: "-")
                        - marker: marker for the line plot (default: "")
                        - xlabel: x-axis label (default: "Precipitation Bins")
                        - plt_title: title of the plot (default: "Normalized Rank Histogram")
                        - fs: basic font size used in plot labels (default: 14)
    """
    # get plot parameters
    figsize = plt_kwargs.pop("figsize", (9, 6))
    lc = plt_kwargs.pop("line_color", "blue")
    ls = plt_kwargs.pop("linestyle", "-")
    marker = plt_kwargs.pop("marker", "")
    xlabel = plt_kwargs.pop("xlabel", "Precipitation Bins")
    plt_title = plt_kwargs.pop("plt_title", 'Normalized Rank Histogram')
    fs = plt_kwargs.pop("fs", 14)
    
    # Plot normalized histogram as a line plot
    bin_centers = np.arange(rank_norm.size)/rank_norm.size
    nranks = rank_norm.shape[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(bin_centers, rank_norm.values, marker=marker, linestyle=ls, color=lc, label=plt_title)
    # plot reference line
    ax.plot(bin_centers, np.repeat(1./nranks, nranks) , marker="", linestyle="--", color="green")
    
    # Labels and title
    ax.set_xlabel(xlabel, fontsize=fs-2)
    ax.set_ylabel('Normalized Frequency', fontsize=fs-2)
    ax.set_title(plt_title, fontsize=fs)
    ax.set_xticks(np.linspace(0, 1., 11))  # Ensure ticks match rank categories
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend()

    if lshow:
        plt.show()

    # save plot and close figure
    plt_fname = Path(plt_fname)
    plt_fname = plt_fname.with_suffix(".png") if not plt_fname.suffix == ".png" else plt_fname
    print(f"Save plot in file '{plt_fname}'")
    fig.savefig(plt_fname, bbox_inches="tight")