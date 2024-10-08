import matplotlib.pyplot as plt
import itertools as it
import pathlib as pl

from utils.read_atmorep_data import HandleAtmoRepData, HandleAtmoRepDataDask
import datetime as dt

MODEL_ID = "cp73kj9o"
VARNAME = "specific_humidity"
RESULTS_BASEDIR = pl.Path("/p/scratch/deepacf/atmorep/augmented_results/")

def plot_quality_controll(time_chunk):
    lead_time = time_chunk.datetime.size
    n_lvls = time_chunk.ml.size
    cols = [time.astype("datetime64[h]") for time in time_chunk.datetime.values]

    fig, axs = plt.subplots(5, 6, sharex=True, sharey=True)
    fig.set_size_inches((8.1, 4))
    pad = 5  # in points
    for ax, col in zip(axs[0], cols):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="x-small",
            ha="center",
            va="baseline",
        )

    for time_step, ml_idx in it.product(range(lead_time), range(n_lvls)):
        ax = axs[ml_idx, time_step]

        global_data = time_chunk.isel(datetime=time_step, ml=ml_idx)
        ax.imshow(global_data)
        ax.set_axis_off()

    fig.tight_layout(w_pad=0.1, h_pad=0.1)

print("starting")
data_load_start = dt.datetime.now()
data_loader = HandleAtmoRepData(MODEL_ID, results_basedir=RESULTS_BASEDIR)
result_data = data_loader.read_data(VARNAME, "pred")
print("data load time: ", dt.datetime.now() - data_load_start)

result_data = result_data[0]

print(result_data.shape, result_data.chunks)

calc_mean_start = dt.datetime.now()
result = result_data.mean().compute()
print("result:", result)
print("calculation time global mean: ", dt.datetime.now() - calc_mean_start)

print("beginn plotting:")
plot_quality_controll(result_data.isel(datetime=range(6)))
plt.savefig(RESULTS_BASEDIR / "quality_controll_global_map")
print(f"saved plots at: {RESULTS_BASEDIR}")