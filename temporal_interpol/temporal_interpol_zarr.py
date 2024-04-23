import zarr
import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import Scores, calc_scores_item
from utils.plotter import Plotter
from utils.read_atmorep_data import HandleAtmoRepData
from utils.utils import get_pl_level, grib_index, get_units
from forecast.forecasting_utils import get_aimodels_scores

model_id  = "cizag5co"
field = "vorticity"
level = 123
input_dir = "/p/scratch/atmo-rep/results/"
out_dir   = "./figures/forecast/"
metrics = ["rmse"]
ar_data = HandleAtmoRepData(model_id, input_dir)
da_target  = ar_data.read_data(field, "target", ml = level)
da_pred    = ar_data.read_data(field, "pred", ml = level)
da_source  = ar_data.read_data(field, "source", ml = level)
atmorep_scores = [calc_scores_item(pred, target, metrics, "all") for pred, target in zip(da_pred, da_target)]
atmorep_scores = np.array(atmorep_scores)

print("AtmoRep")
for m_idx, m in enumerate(metrics):
    print(m, np.mean(atmorep_scores[:, m_idx]))

print("Lineal Interpolation")
da_interpol = []
da_target_intrp = []
for tidx in range(len(da_target)): 

    times = [ t for t in np.unique(da_source[tidx].datetime) if t not in np.unique(da_target[tidx].datetime) ]
    interpol   = da_source[tidx].sel(datetime = times).sortby("lat")
    interpol   = interpol.interp(datetime=np.unique(da_target[tidx].datetime))
    da_interpol.append(interpol)
    #TO-DO: use da_target instead of recreating a new one. 
    #This is simpler because of shape/dimensions differences between source and target
    da_target_intrp.append(da_source[tidx].sel(datetime = np.unique(da_target[tidx].datetime)).sortby("lat"))

interpol_scores = [calc_scores_item(inter, target, metrics, "all") for inter, target in zip(da_interpol, da_target_intrp)]
interpol_scores = np.array(interpol_scores)
width = 0.3
for m_idx, m in enumerate(metrics):
    atmorep = np.mean(atmorep_scores[:, m_idx])
    interpol = np.mean(interpol_scores[:, m_idx])
    print("AtmoRep", m, atmorep)
    print("Lin. Interpol.", m, interpol) 

    fig, axs = plt.subplots(figsize= (12, 5))
    plt.title(f"Model Level: {level}" )
    axs.barh( [1-width], atmorep, width-0.02, align='edge', color = "royalblue",
                error_kw=dict(lw=0.5, capsize=1, capthick=0.5), label = "AtmoRep")
    axs.barh( [1], interpol, width-0.02, align='edge', color = "orange",
                error_kw=dict(lw=0.5, capsize=1, capthick=0.5), label = 'lin. interpol.')   

    axs.spines[['right', 'top']].set_visible(False)
    axs.set_xscale('log')
    axs.set_xlabel(f"{m} [{get_units(field)}]")      
    
    for c in axs.containers:
        axs.bar_label(c, fmt="%.2e", padding = 3)
    
    axs.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(f"./temporal_interpol_id{model_id}_{field}_{m}_{level}.png")
    plt.close()
                  
