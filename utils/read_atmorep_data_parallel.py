import zarr
import numpy as np
import xarray as xr
import pathlib as pl
import collections
import functools


Sample = collections.namedtuple("Sample", ["coords", "data"])


class Samples:
    def __init__(self, path: pl.Path, field: str):
        self.sample_key_prefix = "sample="
        self.sample_idx_format = r"{:05d}"
        store = zarr.ZipStore(path)
        self.samples = zarr.group(store)[field]
        example_sample = self.get_sample(0)
        self.size = len(self.samples)
        self.lead_time = example_sample.coords["datetime"].size

    def get_sample_idxs(self, chunk_idx: int):
        return self.sample_idxs[self.inverse == chunk_idx]

    def as_key(self, sample_idx: int):
        return self.sample_key_prefix + self.sample_idx_format.format(sample_idx)

    def get_index(self, key: str):
        return int(key.removeprefix(self.sample_key_prefix))

    def get_sample(self, idx) -> Sample:
        sample = self.samples[self.as_key(idx)]
        coords = {key: values for key, values in sample.arrays() if not key == "data"}

        return Sample(coords, sample["data"])


class ChunkedData:
    def __init__(self, samples):
        self.samples = samples
        self.lead_time = self.samples.lead_time
        self.time_chunks, self._chunk_to_samples = self.from_samples(self.samples)
        self._samples_idxs = np.arange(self._chunk_to_samples.size)
        self._forecast_times = self.time_chunks[:, -1]

        self.global_coords = self.get_global_coords()
        self.dims = ["datetime", "ml", "lat", "lon"]

        self.shape = [self.global_coords[dim].size for dim in self.dims]
        self.chunks = [
            self.lead_time if dim == "datetime" else dim_size
            for dim_size, dim in zip(self.shape, self.dims)
        ]

    def get_global_coords(self):
        example_sample = self.samples.get_sample(0)
        
        print(self._forecast_times.size, self._forecast_times.min(), self._forecast_times.max())

        start = self._forecast_times.min() - self.lead_time
        end = self._forecast_times.max()
        times = np.arange(start, end, np.timedelta64(1, "h"), dtype="datetime64[ns]")
        times += np.timedelta64(1, "h")
        dx = np.abs(example_sample.coords["lon"][1] - example_sample.coords["lon"][0])
        dy = np.abs(example_sample.coords["lat"][1] - example_sample.coords["lat"][0])
        lats = np.linspace(-90.0, 90.0, num=int(180 / dy) + 1, endpoint=True)
        lons = np.linspace(0, 360, num=int(360 / dx), endpoint=False)
        levels = example_sample.coords["ml"]
        return {"datetime": times, "ml": levels, "lat": lats, "lon": lons}

    @classmethod
    def from_samples(cls, samples: Samples):
        index_datetimes = (
            (samples.get_index(key), sample.datetime)
            for key, sample in samples.samples.groups()
        )
        sample_times = np.empty(
            shape=(samples.size, samples.lead_time),
            dtype=("datetime64[ns]"),
        )
        for idx, datetimes in index_datetimes:
            sample_times[idx] = datetimes

        return np.unique(sample_times, return_inverse=True, axis=0)

    def load_chunk(self, chunk: xr.DataArray) -> xr.DataArray:
        forecast_time = chunk["datetime"].values[-1]
        try:
            for sample in self.get_samples(forecast_time):
                chunk.loc[sample.coords] = sample.data
        except ValueError:  # no data for this chunk
            # chunk.loc[{"datetime": chunk["datetime"]}] = np.nan
            pass

        return chunk

    def get_samples(self, forecast_time: np.datetime64):
        chunk_idx = self.get_chunk_idx(forecast_time)
        print(forecast_time, chunk_idx)
        return self._get_chunk_samples(chunk_idx)

    def get_chunk_idx(self, forecast_time: np.datetime64):
        # assume forecasttimes are unique
        chunk_idx = np.argwhere(self._forecast_times == forecast_time)
        if chunk_idx.size == 0:
            msg = f"no data for forecast time {forecast_time}"
            raise ValueError(msg)
        return int(chunk_idx)

    def _get_chunk_samples(self, chunk_idx: int):
        sample_idxs = self._get_samples_idxs(chunk_idx)
        return [self.samples.get_sample(idx) for idx in sample_idxs]

    @functools.cache
    def _get_samples_idxs(self, chunk_idx: int):
        return self._samples_idxs[self._chunk_to_samples == chunk_idx]