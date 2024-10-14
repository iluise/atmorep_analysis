import zarr
import numpy as np
import xarray as xr
import pathlib as pl
import collections
import functools
import os
import typing
import dataclasses
import itertools as it


@dataclasses.dataclass
class Sample:
    coords: dict[str, zarr.Array]
    data: zarr.Array

IndexRange = collections.namedtuple("IndexRange", ["start", "end"])


class Samples:
    def __init__(self, path: pl.Path, field: str):
        self.sample_key_prefix = "sample="
        self.sample_idx_format = r"{:05d}"
        
        store = zarr.ZipStore(path)
        self.samples = zarr.group(store)[field]

        example_sample = self.get_sample(0) # assume sample idx 0 always present
        self.dims = list(example_sample.coords.keys())
        self.shape = [example_sample.coords[dim].size for dim in self.dims]
        self.size = len(self.samples)
        self.lead_time = example_sample.coords["datetime"].size
        self.dx = np.abs(
            example_sample.coords["lon"][1] - example_sample.coords["lon"][0]
        )
        self.dy = np.abs(
            example_sample.coords["lat"][1] - example_sample.coords["lat"][0]
        )
        self.levels = example_sample.coords["ml"]

    def get_sample_idxs(self, chunk_idx: int):
        return self.sample_idxs[self.inverse == chunk_idx]

    def as_key(self, sample_idx: int):
        return self.sample_key_prefix + self.sample_idx_format.format(sample_idx)

    def get_index(self, key: str):
        return int(key.removeprefix(self.sample_key_prefix))

    def get_sample(self, idx) -> Sample:
        sample = self.samples[self.as_key(idx)]
        coords = {
            "datetime": sample["datetime"],
            "ml": sample["ml"],
            "lat": sample["lat"],
            "lon": sample["lon"]
        }

        return Sample(coords, sample["data"])


class EnsembleSamples:
    def get_sample(self, idx) -> Sample:
        sample = self.samples[self.as_key(idx)]
        coords = {
            "ens": sample["ens"], # add additional dimension
            "datetime": sample["datetime"],
            "ml": sample["ml"],
            "lat": sample["lat"],
            "lon": sample["lon"],
        }

        return Sample(coords, sample["data"])


class ChunkedData:
    def __init__(self, samples, dask_chunk_factor=1):
        self.samples = samples
        self.time_chunks, self._chunk_to_samples = self.from_samples(self.samples)
        self._samples_idxs = np.arange(self._chunk_to_samples.size)
        self._forecast_times = self.time_chunks[:, -1]

        self.lead_time = self.samples.lead_time
        self.dims = self.samples.dims
        self.dy, self.dx = self.samples.dy, self.samples.dx

        example_sample = self.samples.get_sample(0)

        self.global_coords = self.get_global_coords()

        self.shape = [self.global_coords[dim].size for dim in self.dims]
        self.chunks = [
            self.lead_time*dask_chunk_factor if dim == "datetime" else dim_size
            for dim_size, dim in zip(self.shape, self.dims)
        ]

        self._lat_padding = self.samples.shape[3] # use one entire sample as padding

    def get_global_coords(self):
        start = self._forecast_times.min() - np.timedelta64(self.lead_time, "h")
        end = self._forecast_times.max()

        times = np.arange(start, end, np.timedelta64(1, "h"), dtype="datetime64[ns]")
        times += np.timedelta64(1, "h")

        lats = np.linspace(-90.0, 90.0, num=int(180 / self.dy) + 1, endpoint=True)
        lons = np.linspace(0, 360, num=int(360 / self.samples.dx), endpoint=False)

        return {"datetime": times, "ml": self.samples.levels, "lat": lats, "lon": lons}

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

        # add padding to lat dim to account for wrap around of indexes/coords
        lat_size = chunk.shape[-1]
        lat_size_padded = lat_size + self._lat_padding
        shape_with_padding = (*chunk.shape[:-1], lat_size_padded)
        buffer = np.empty(shape_with_padding)

        try:
            for sample in self.get_samples(forecast_time):
                lats, lons = self.coords_as_ranges(sample.coords)
                # conform data: swap ml/datetime axis, flip lat axis
                sample_data = np.flip(np.swapaxes(sample.data, 0, 1), axis=2)
                buffer[:, :, lats.start:lats.end, lons.start:lons.end] = sample_data
        except IndexError:
            buffer[:] = np.nan

        return xr.DataArray( # remove padding again
            buffer[:,:,:,:lat_size], coords=chunk.coords, dims=chunk.dims
        )

    def coords_as_ranges(self, coords: dict["str", typing.Any]) -> tuple[IndexRange]:
        global_lats_min = self.global_coords["lat"][0]
        sample_lats_min = coords["lat"][-1]

        lat_size = coords["lat"].size
        # substract global min coord from sample min coord to obtain index
        lat_start_idx = int((sample_lats_min - global_lats_min)*(1/self.dy))
        lat_range = IndexRange(lat_start_idx, lat_start_idx + lat_size)

        global_lons_min = self.global_coords["lon"][0].astype(int)
        sample_lons_min = coords["lon"][0].astype(int)

        lon_size = coords["lon"].size
        lon_start_idx = int((sample_lons_min - global_lons_min)*(1/self.dx))
        lon_range = IndexRange(lon_start_idx, lon_start_idx+lon_size)

        return lat_range, lon_range

    

    def _get_chunk_buffer(self, chunk: xr.DataArray):
        """
        construct DataArray with numpy array as backend array.
        
        Avoids NotImplementedError: xarray can't set arrays with multiple array indices to dask yet, by using a separate buffer instead of reusing "chunk".
        """
        buffer = np.empty(chunk.shape)
        return xr.DataArray(buffer, coords=chunk.coords, dims=chunk.dims)

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

class XarrayIndexing(ChunkedData):
    def load_chunk(self, chunk: xr.DataArray) -> xr.DataArray:
        forecast_time = chunk["datetime"].values[-1]
        buffer_da = self._get_chunk_buffer(chunk)
        try:
            for sample in self.get_samples(forecast_time):
                buffer_da.loc[sample.coords] = np.swapaxes(sample.data, 0, 1)
        except ValueError:  # no data for this chunk
            buffer_da.loc[{"datetime": chunk["datetime"]}] = np.nan

        return buffer_da


class IndexLookup(ChunkedData):    
    def load_chunk(self, chunk: xr.DataArray) -> xr.DataArray:
        forecast_time = chunk["datetime"].values[-1]
        buffer_da = self._get_chunk_buffer(chunk)
        try:
            for sample in self.get_samples(forecast_time):
                lat_range, lon_range = self.coords_as_ranges(sample.coords)
                buffer_da[:, :, lat_range, lon_range] = np.swapaxes(sample.data, 0, 1)
        except ValueError:  # no data for this chunk
            buffer_da.loc[{"datetime": chunk["datetime"]}] = np.nan

        return buffer_da


class Baseline(ChunkedData):
    def load_chunk(self, chunk: xr.DataArray) -> xr.DataArray:
        n_samples = chunk.datetime.size * 196
        da_list = []
        coords = {}
        time_coords = chunk.datetime.values
        buffer_da = self._get_chunk_buffer(chunk)

        # load any samples equivalent to one chunk
        for sample in it.islice(self.samples.samples, 1, n_samples + 1):
            da_list.append(self.read_chunk_baseline(sample, coords, time_coords))

        self.merge_chunks_baseline(buffer_da, da_list)

        return buffer_da

    def read_chunk_baseline(self, sample: zarr.Group, coords, time_coords):
        coords.update(
            {
                dim: (
                    self.samples.samples[os.path.join(sample, dim)]
                    if not dim == "datetime"
                    else time_coords
                )
                for dim in self.dims
            }
        )
        data = self.samples.samples[os.path.join(sample, "data")]
        # print(data.shape, self.dims, coords)
        sample_da = xr.DataArray(
            np.swapaxes(data, 0, 1),
            coords=coords,
            dims=self.dims,
            name=f"specific_humidity_{sample.replace('=', '')}",
        )
        return sample_da

    def merge_chunks_baseline(
        self, global_data: xr.DataArray, da_list: list[zarr.Group]
    ):
        for sample in da_list:
            global_data.loc[
                {
                    "datetime": sample["datetime"],
                    "lat": sample["lat"],
                    "lon": sample["lon"],
                }
            ] = sample
