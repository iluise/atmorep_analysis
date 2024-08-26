import os
import json
from typing import List
import zarr
from random import randint
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import xarray as xr
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import cartopy
import cartopy.crs as ccrs  #https://scitools.org.uk/cartopy/docs/latest/installing.html
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

from utils.utils import get_units, field_full_name

class Plotter(object):
    """
    Contains all basic plotting functions.
    """

    def __init__(self, field: str, model_id: str, out_dir: str, level : str):
         """
        :param field: plotted variable  
        :param model_id: ID of Atmorep-run to process
        :param out_dir: output folder where plots are saved
         :param level: vertical level
         """
         self.field    = field
         self.model_id = model_id if model_id.startswith("id") else f"id{model_id}"
         self.out_dir  = out_dir
         self.level    = level
         
######################################################

    def CustomPalette(self):
        """
        function to build a Mathematica-like palette
        """
        colors = [ (0.278, 0.380, 0.620), (0.867, 0.647, 0.365), (0.991, 0.949, 0.765)]
        cmap = mcolors.LinearSegmentedColormap.from_list('YlBu', colors, N=100)
        return cmap
    
######################################################

    def plot(self, data, label = "_", name = None, ax = None, bins = 50, log_yscale = False, xlabel = None, ylabel = None):
        """
        1D plotting of multiple lists of arrays
        :param data: can be a list of samples for a single histogram, or a dictionary with multiple histograms and labels.
        data = [sample for sample in data] --> single dataset to be plotted
        data = {"target" : data_target, "prediction" : data_pred} --> multiple datasets
        :param name: name of the plot used in savefig 
        :param bins: number of bins
        :param avg_dims: dimensions to average on before plotting 
        """
        if data == None:
            return

        begin = datetime.datetime.now()
        save_fig = False
        if ax == None:
            fig, ax = plt.subplots()
            save_fig = True

        assert type(data) == dict or  type(data) == list, "Info::Plotter::plot - Data must be a dictionary or a list."
        
        log_data = data if type(data) is dict else {label : data}

        #set limits from the first item in the dictionary:
        data_min = float(min([np.amin(d) for d in next(iter(log_data.values()))]))
        data_max = float(max([np.amax(d) for d in next(iter(log_data.values()))]))
        edges = np.linspace(data_min, data_max, bins)

        for ld_label, ld in log_data.items():
            edges = np.linspace(data_min, data_max, bins)
            hist = [np.histogram(d, bins = bins, range = [data_min, data_max])[0] for d in ld]
            #transform into a numpy array
            hist = np.sum(np.array(hist, dtype = np.float32), axis = 0) #sum over all histograms = stack
            ax.plot(edges,hist,label = ld_label, linewidth = 0.5)
                
        ax.set_xlabel(f"{self.field} [{get_units(self.field)}]" if xlabel == None else xlabel)
        ax.set_ylabel("counts"  if xlabel == None else ylabel)

        if sum(1 for k in log_data.keys() if k != "_") > 1:
            ax.legend(frameon = False)

        if log_yscale:
            ax.set_yscale('log')

        print(datetime.datetime.now() - begin)
        
        if save_fig:
            fig.savefig(f"{self.out_dir}/plot_{name}_{self.field}_{self.model_id}_ml{self.level}.png")
            fig.savefig(f"{self.out_dir}/plot_{name}_{self.field}_{self.model_id}_ml{self.level}.pdf")       
            plt.close()
        else:
            return ax

######################################################

    def plot_along_dim(self, data , dim: str, name: str, label = "_", bins = 50, log_yscale = True, single_plot = True):
        """
        1D plotting of multiple lists of arrays looping along a specific dimension
        :param data: can be a list of samples for a single histogram, or a dictionary with multiple histograms and labels.
          list:   data = [sample for sample in data] --> single dataset to be plotted
          dict:   data = {"target" : data_target, "prediction" : data_pred} --> multiple datasets
        :param name: name of the plot used in savefig
        :param bins: number of bins
        :param log_yscale: set y axis to logaithmic scale
        :param single_plot: plot everything on a single plot or each sub-set on separate subplots.
        """
        assert type(data) == dict or type(data) == list, "Info::Plotter::plot_along_dim - Data must be a dictionary or a list."

        log_data = data if type(data) is dict else {label : data}
        loop_values = next(iter(log_data.values()))[0][dim].values #get values along which we will split the dataset
        
        #Check that the other datasets have exactly the same values along that dimension.
        if len(log_data.keys()) > 1:
            for da_name, da in log_data.items():
                check = np.unique([d[dim].values for d in da])
                if not (check == loop_values).all():
                    print(f"Error:: Detected different values. Skipping dim = {dim}")
                    return

        if (single_plot):
            log_data_sel = {}
            for value in loop_values:
                for n, da in log_data.items():
                    da_sel = [d.sel({dim:value}) for d in da]
                    log_data_sel[f"{n} {dim}={value}"] = da_sel
            self.plot(log_data_sel, name = name, log_yscale = log_yscale)
                
        else:
            fig, ax = plt.subplots(len(loop_values), figsize = (15, 8), sharex = True)
            if len(loop_values) > 3:
                fig, ax = plt.subplots(2, len(loop_values) % 2, sharex = True)

            for value, ax_i in zip(loop_values, ax.ravel()):
                log_data_sel = {}
                for n, da in log_data.items():
                    da_sel = [d.sel({dim:value}) for d in da]
                    log_data_sel[n] = da_sel
                ax_i = self.plot(log_data_sel, ax = ax_i, log_yscale = log_yscale)
                ax_i.set_title(f"dimension {dim} = {value}")
            
            fig.tight_layout()
            fig.savefig(f"{self.out_dir}/plot_{name}_{self.field}_{self.model_id}_ml{self.level}.png")
            plt.close()
              
######################################################

    def mapshow(self, data: list, name: str, show_contours = False, res = 0.25, cmap = "PuBuGn", colorbar = True, ax = None):
        """
        2D plotting of a list of arrays 
        :param data: a list of samples 
          list:   data = [sample for sample in data] --> single dataset to be plotted
        :param name: name of the plot used in savefig
        :param bins: number of bins
        :param log_yscale: set y axis to logaithmic scale
        :param single_plot: plot everything on a single plot or each sub-set on separate subplots.
        """
        begin = datetime.datetime.now()
        
        x_min = float(min([np.amin(d.lat) for d in data]))
        x_max = float(max([np.amax(d.lat) for d in data]))
        edges_x = np.arange(x_min, x_max+res, res)
        
        y_min = float(min([np.amin(d.lon) for d in data]))
        y_max = float(max([np.amax(d.lon) for d in data]))
        edges_y = np.arange(y_min, y_max+res, res)

        hist = np.zeros([len(edges_x), len(edges_y)])
        hist_counts = np.zeros([len(edges_x), len(edges_y)])

        save_fig = True if ax == None else False

        assert len(data[0][0].values.shape) < 3, "Plotter::Mapshow - Please select a time-step."
                    
        for it, sample in enumerate(data):
            if it % 100 == 0:
                print(f"Info:: Plotter::mapshow - running sample {it}/{len(data)}", end = '\r', flush = True)
            for da in sample:
                x_idx, y_idx = np.meshgrid(np.floor((x_max - da.lat.values)/res).astype(int), np.floor((da.lon.values-y_min)/res).astype(int))
                hist[x_idx, y_idx] += da.values
                hist_counts[x_idx, y_idx] += 1
            
        print(datetime.datetime.now() - begin)

        hist /= hist_counts
        
        if show_contours: #show axis lines and countries
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180)) if ax == None else ax

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
            im = plt.imshow(hist, cmap=cmap, extent=[-180,180,-90,90])
        else:
            fig, ax = plt.subplots() if ax == None else (None, ax)
            im = plt.imshow(hist, cmap=cmap)

        if colorbar:
            cbar = plt.colorbar(im, shrink=0.7) 
            cbar.set_label(get_units(self.field), y=-0.04, ha='right', rotation=0)

        if save_fig:
            fig.tight_layout()
            fig.savefig(f"{self.out_dir}/map_{name}_{self.field}_{self.model_id}_ml{self.level}.png")
            plt.close()
        else:
            return ax

######################################################

    def scatter_plot(self, data1: list, data2: list, name: str,  bins = [50, 50], xlabel = "", ylabel = ""):
        xmin = float(min([np.amin(d) for d in data1]))
        ymin = float(min([np.amin(d) for d in data2]))
        xmax = float(min([np.amax(d) for d in data1]))
        ymax = float(min([np.amax(d) for d in data2]))

        begin = datetime.datetime.now()
        
        assert len(data1) == len(data2), f"Error:: Scatter Plot - dimensions do not match. {len(data1)} {len(data2)}."

        hist = [np.histogram2d(d1.values.flatten(), d2.values.flatten(), bins = bins, range = [[xmin, xmax], [ymin, ymax]])[0] for d1, d2 in zip(data1, data2)]
        
        #transform into a numpy array
        hist = np.sum(np.array(hist, dtype = np.float32), axis = 0) #sum over all histograms = stack

        fig_std = plt.figure(figsize=(8, 6))
        im = plt.imshow(hist, origin='lower')
        plt.colorbar(im,shrink=0.7)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig( f"{self.out_dir}/scatter_{name}_{self.field}_{self.model_id}_ml{self.level}.png")
        plt.close()
        print(datetime.datetime.now() - begin)
        
######################################################

    def hist(self, data , label = "_", name = None, ax = None, bins = 50, log_yscale = False, xlabel = None, ylabel = None, range = None ):
        """
        1D plotting of multiple lists of arrays
        :param data: can be a list of samples for a single histogram, or a dictionary with multiple histograms and labels.
          list:   data = [sample for sample in data] --> single dataset to be plotted
          dict:   data = {"target" : data_target, "prediction" : data_pred} --> multiple datasets
        :param name: name of the plot used in savefig
        :param bins: number of bins
        :param avg_dims: dimensions to average on before plotting
        """
        if data == None:
            return
        
        begin = datetime.datetime.now()
        save_fig = True if ax == None else False
        if ax == None:
            fig, ax = plt.subplots() 

        assert type(data) == dict or  type(data) == list, "Info::Plotter::plot - Data must be a dictionary or a list."
        log_data = data if type(data) is dict else {label : data}
        
        data_min = float(min([np.amin(d) for d in next(iter(log_data.values()))])) if range[0] == None else range[0]
        data_max = float(max([np.amax(d) for d in next(iter(log_data.values()))])) if range[1] == None else range[1]
        
#        data_min = float(min([np.amin(d) for d in log_datasets[0][1]])) if range[0] == None else range[0]
#        data_max = float(max([np.amax(d) for d in log_datasets[0][1]])) if range[1] == None else range[1]
        edges = np.linspace(data_min, data_max, bins)

        for ld_name, ld in log_data.items():
            edges = np.linspace(data_min, data_max+(data_max-data_min)/bins, bins+1)
            hist = [np.histogram(d, bins = bins, range = [data_min, data_max])[0] for d in ld]

            #transform into a numpy array
            hist = np.sum(np.array(hist, dtype = np.float32), axis = 0) #sum over all histograms = stack
            ax.stairs(hist, edges, label = ld_name, linewidth = 0.5, color = "blue")

        ax.set_xlabel(f"{self.field} [{get_units(self.field)}]" if xlabel == None else xlabel)
        ax.set_ylabel("counts"  if xlabel == None else ylabel)

        #legend
        if sum(1 for k in log_data.keys() if k != "_") > 1:
            ax.legend(frameon = False)

        #log scale
        if log_yscale:
            ax.set_yscale('log')

        print(datetime.datetime.now() - begin)

        if save_fig:
            fig.savefig(f"{self.out_dir}/plot_{name}_{self.field}_{self.model_id}_ml{self.level}.png")
            plt.close()
        else:
            return ax

######################################################

    def graph(self, data, x_val: list, name: str, label = "_", ax = None, std_dev = False, colors = None, ylabel = "", xlabel = ""):
        """
        1D plotting of multiple lists of arrays along the same x axis. the values along the xaxis are specified by x_val
        :param data: can be a list of samples for a single histogram, or a dictionary with multiple histograms and labels.
          list:   data = [sample for sample in data] --> single dataset to be plotted
          dict:   data = {"target" : data_target, "prediction" : data_pred} --> multiple datasets
        :param name: name of the plot used in savefig
        :param std_dev: average the first dimension and plot it as error associated to the measurement. Applications: e.g. forecasting. 
        :param colors: list of colors of the same length of log_dataset. 
        :param ylabel: label on the y-axis
        :param xlabel: label on the x-axis
        """
        
        assert type(data) == dict or  type(data) == list, "Info::Plotter::plot - Data must be a dictionary or a list."
        log_data = data if type(data) is dict else {label : data}

        if colors == None: #random colors
            colors = ['#%06X' % randint(0, 0xFFFFFF) for _ in range(len(log_data.keys()))]

        assert (len(colors) <= len(log_data.keys())), f"Length of colors and dataset do not match. Please verify. colors: {len(colors)}, data: {len(log_data.keys())}"
        
        save_fig = True if ax == None else False
        if ax == None:
            fig, ax = plt.subplots()
            
        for idx, (ld_name, ld) in enumerate(log_data.items()): #graphs
            if std_dev:
                data_mean = np.mean(ld, axis = 0)
                data_std = np.std(ld, axis = 0)
                plt.fill_between(x_val, data_mean - data_std, data_mean + data_std, color = colors[idx], alpha=0.1, label='_') 
            else:
                data_mean = ld
            plt.plot(x_val, data_mean, marker='.', linestyle='-', alpha = 1, linewidth=0.5, label = label, color = colors[idx])

        if sum(1 for k in log_data.keys() if k != "_") > 1:
            plt.legend(frameon = False)

        plt.tight_layout()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if save_fig:
            fig.savefig(f"{self.out_dir}/plot_{name}_{self.field}_{self.model_id}_ml{self.level}.png")
        else:
            return ax

        plt.close()

######################################################

    def profile(self, data: list, name: str, groupby_dim: str, mean_dim: list, bins = 50, ax = None):
        """
        1D plotting of multiple lists of arrays along the same x axis. the values along the xaxis are specified by x_val
        :param data: is a list of samples for a single histogram
          list:   data = [sample for sample in data] --> single dataset to be plotted
          dict:   TO-DO:: support for dictionary
        :param name: name of the plot used in savefig
        :param std_dev: average the first dimension and plot it as error associated to the measurement. Applications: e.g. forecasting.
        :param colors: list of colors of the same length of log_dataset.
        :param ylabel: label on the y-axis
        :param xlabel: label on the x-axis
        """
        
        begin = datetime.datetime.now()
        save_fig = True	if ax == None else False
        if ax == None:
            fig, ax = plt.subplots()

        prof = [d.groupby(groupby_dim).mean().mean(mean_dim) for d in data]
        data_min = float(min([np.amin(d[groupby_dim]) for d in prof]))
        data_max = float(max([np.amax(d[groupby_dim]) for d in prof]))
        edgesx = np.linspace(data_min, data_max, bins)
        hist = [np.histogram(d[groupby_dim], bins = edgesx, weights = d.values)[0] for d in prof]
        hist_counts = [np.histogram(d[groupby_dim], bins = edgesx)[0] for d in prof]
        hist = np.sum(np.array(hist, dtype = np.float32), axis = 0)
        hist_counts = np.sum(np.array(hist_counts, dtype = np.float32), axis = 0)
        hist = hist / hist_counts
        ax.set_xlabel(groupby_dim)
        ax.set_ylabel(name)
        ax.plot(edgesx[:-1],hist, linewidth = 0.5)
        print(datetime.datetime.now() - begin)
        if save_fig:
            fig.savefig(f"{self.out_dir}/profile_{name}_{self.field}_{self.model_id}_ml{self.level}.png")
        else:
            return ax
        
        plt.close()
