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

import numpy as np
import xarray as xr
import properscoring as ps
import pandas as pd

from utils.utils import *
from utils.metrics import *
from utils.plotting import *
from datetime import datetime

import netCDF4
import cfgrib

# plotting
import seaborn as sns # for data visualization
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 0.1
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.colors as colors

for field in ["velocity_u", "velocity_v", "velocity_z", "specific_humidity", "temperature"]:
    for month in range(1, 13):
        for lvl in [1000, 925, 850, 700, 500]:
            mean = get_climatological_mean(field, month, lvl)
            if (mean.mean() ==0) and (mean.std() == 0):
                mean = compute_climatological_mean(field, month, lvl)
                
