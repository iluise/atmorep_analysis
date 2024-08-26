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

import pdb
import zarr
import matplotlib.pyplot as plt

from utils.metrics import Scores
from utils.read_atmorep_data import HandleAtmoRepData

#model_id1, label1 = '7w7rh21c', "weightMSE_stats"
#model_id2, label2 = 'dtqm59uy', "MSE_stats"
#field = 'velocity_u'

#model_id1, label1 = 'vuk4jkmf', "weightMSE_stats"
#model_id2, label2 = 'acsvut65', "MSE_stats"
#field = 'temperature'

model_id1, label1 = 'lld5w50x', '3x9x9 old config'
model_id2, label2 = 'd018p5pk', '3x18x18 global norm'
field = 'specific_humidity'

metrics = ['rmse'] #, 'rmse'] 
levels = [114]
# #MSE
ar_data1 = HandleAtmoRepData(model_id1, "./results/")
pred1   = ar_data1.read_data(field, "pred"  , ml = levels)
target1 = ar_data1.read_data(field, "target", ml = levels)
ens1    = ar_data1.read_data(field, "ens"   , ml = levels)
score_engine1 = Scores(pred1, target1, ens1, avg_dims = [])

# for i in range(1, 16):
#   (ens1[0,0,0] - ens1[i,0,0]).plot(cmap ='RdBu')
#   plt.savefig(f"ens0-ens{i}_{field}_{label1}.png")
#   plt.close()

for metric_name in metrics:
  metric1 = score_engine1(metric_name)
  metric1[0,0].plot()
  plt.savefig(f"{metric_name}_{field}_{label1}.png")
  plt.close()
  metric1.plot(bins = 50)
  plt.savefig(f"{metric_name}_{field}_{label1}_hist.png")
  plt.close()


#CRPS
ar_data2 = HandleAtmoRepData(model_id2, "./results/")
pred2   = ar_data2.read_data(field, "pred"  , ml = levels)
target2 = ar_data2.read_data(field, "target", ml = levels)
ens2    = ar_data2.read_data(field, "ens"   , ml = levels)
score_engine2 = Scores(pred2, target2, ens2, avg_dims = [])

# for i in range(1, 16):
#   (ens2[0,0,0] - ens2[i,0,0]).plot(cmap ='RdBu')
#   plt.savefig(f"ens0-ens{i}_{field}_{label2}.png")
#   plt.close()

for metric_name in metrics:
  metric2 = score_engine2(metric_name)
  metric2[0,0].plot()
  plt.savefig(f"{metric_name}_{field}_{label2}.png")
  plt.close()
  metric2.plot(bins = 50)
  plt.savefig(f"{metric_name}_{field}_{label2}_hist.png")
  plt.close()