#
#  Copyright (C) 2023
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
#
#  description : Code to plot the training diagnostic routine.
#                It monitors target, preds, source and the ensemble.
#
#  license     : MIT licence
#
####################################################################################################

import zarr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import calendar
import random
import os

from utils.metrics import Scores, calc_scores_item
from utils.plotter import Plotter
from utils.read_atmorep_data import HandleAtmoRepData
from trainings.training_utils import *

#models  = {'iv17yhus': "pre-trained", 'tj8vsp65' : "12,6,12 x 3,9,9 local norm", "10iyls22": "12,6,12 x 3,9,9 global norm", "heud0haf": "12,3,6 x 13,18,18 global norm", "0u8e7tbg" : "train continue"}  #'idmi4tsq5d'
#models  = {'9hvndnjq': "pre-trained", 'u2k4mvxq' : "12,6,12 x 3,9,9 local norm", "a7v8m62x": "12,6,12 x 3,9,9 global norm", "v2i1tbok": "12,3,6 x 13,18,18 global norm", "f4hfzcjf" : "continue-training"}  #'idmi4tsq5d'
models  = {'9ij898wm': "pre-trained", '7gacxauw' : "12,6,12 x 3,9,9 local norm", "a7v8m62x": "12,6,12 x 3,9,9 global norm", "v2i1tbok": "12,3,6 x 13,18,18 global norm", "f4hfzcjf" : "continue-training"}  #'idmi4tsq5d'

field     ="temperature" #"velocity_u" #
levels    = [137, 114, 96, 123, 105] #[137, 123] 
input_dir = "./results/"

out_dir   = f"./figures/"
if not os.path.exists(out_dir):
    out_dir = f"{out_dir}/" if out_dir[-1] != "/" else out_dir
    os.makedirs(out_dir)

for level in levels:
    da_pred = {}
    da_target = {}
    plotter = Plotter(field, "", out_dir, level)
    for model_id, label in models.items():
        ar_data = HandleAtmoRepData(model_id, input_dir)
        # list of xarray DataArrays is returned where each element corresponds to a sample (patch)
        da_pred[f"pred:{label}"]   = ar_data.read_data(field, "pred"  , ml = level)
        da_target[f"target:{label}"] = ar_data.read_data(field, "target", ml = level)

    plotter.plot({**da_pred, **da_target}, name = f"compare_models_ml{level}", log_yscale = True)







