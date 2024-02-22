This project collects examples of analysis scripts for each application supported by AtmoRep. Please refer to the [model page](https://github.com/clessig/atmorep/tree/main) for instructions on how to run the model. 

### File format

The output of the evaluation step is a set of `.zarr` files. Example:
- `model_idc96xrbip.json`  = model settings used in the evaluation phase
- `results_idc96xrbip_epoch00000_pred.zarr`    = file storing all predicted tokens
- `results_idc96xrbip_epoch00000_target.zarr`  = file storing the masked target tokens (aka ground truth) in the same format as predictions
- `results_idc96xrbip_epoch00000_source.zarr` = file storing all the loaded tokens (masked tokens are stored as zeros)
- `results_idc96xrbip_epoch00000_ens.zarr`       = file storing the ensemble predictions
- (optional) `results_idc96xrbip_epoch00000_attention.zarr`  = file storing the attention scores. Written only if `attention = True` at evaluation stage. 

### AtmoRep data interface
The `read_atmorep_data.py` package contains all the functionalities to read the AtmoRep data. The output of read_data is a set of xarrays. 
```
from utils.read_atmorep_data import HandleAtmoRepData

ar_data    = HandleAtmoRepData(model_id, input_dir)
da_target  = ar_data.read_data(field, "target", ml = levels)
da_pred    = ar_data.read_data(field, "pred"  , ml = levels)
```
the output is interfaced with the `metrics.py` package, which contains a set of common metrics to evaluate the model performance. 

```
from utils.metrics import Scores, calc_scores_item
metrics   = ["rmse", "acc", "l1", "bias"]
atmorep_scores = calc_scores_item(da_pred, da_target, metrics, avg = ["lat", "lon"])
```
in this case the scores will be averaged over latitude and longitude. Please use `avg = None` to avoid averaging or `avg = all` if you want to average over all dimensions. 

### Applications
Here below you can find a brief description of the examples available for each application. This is a WIP and new use cases will be added shortly. 

#### Training protocol
This package, contained within the `trainings` folder contains a plotting routine to inspect your trainings. It supports the computation of all the metrics defined within `metrics`:
- 1D plots over local token variables (latitude, longitude, time, local position within the masked area etc..)
- 2D maps averaged over time and per time-step
- plot single token predictions for the 10 examples with the highest and lower _absolute_ and _relative_ errors as well as 50 random examples. 
- analysis of the ensemble distributions (standard deviation, spread vs error etc..). 


#### Weather forecasting
The forecasting analysis is in the `forecasting` folder. This is intended to be an example of how to compare AtmoRep data -- computed with the `global_forecasting` option in `evaluate.py` -- against e.g. PanguWeather data obtained with the [ai-models](https://github.com/ecmwf-lab/ai-models-panguweather/tree/main/ai_models_panguweather) interface provided by ECMWF. 


#### Attention
The folder `attention` contains the code to inspect the attention scores. The values can be obtained by setting `attention = True` at evaluation stage in `evaluate.py`. 
The code 'plot_attention.py' plots the attention maps from the `*_attention.zarr` file. 