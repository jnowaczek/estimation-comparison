#  Copyright (C) 2025 Julian Nowaczek.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
import itertools

import numpy as np
import dask.array as da
import scipy.signal as signal
# noinspection PyProtectedMember
from traitlets import Callable, Int

from estimation_comparison.data_collection.estimator.base import EstimatorBase


class Autocovariance(EstimatorBase):
    block_size = Int(1024)

    def estimate(self, data: np.ndarray) -> da.Array:
        def normalize(block: da.Array) -> np.ndarray:
            print(f"block shape: {block.shape}")
            return da.divide(da.subtract(block, da.mean(block)), da.power(da.std(block), 2))

        def autocorrelate(block):
            return signal.correlate(block, block)

        data_array = da.from_array(data[:data.shape[0] // self.block_size * self.block_size], self.block_size)
        data_array = da.reshape(data_array, (-1, self.block_size))

        normalized = da.apply_along_axis(normalize, 1, arr=data_array)
        numerators = da.apply_along_axis(autocorrelate, 1, arr=normalized)
        denominators = da.apply_along_axis(autocorrelate, 1, arr=normalized)
        a = da.divide(numerators, denominators, out=da.zeros_like(numerators), where=denominators != 0,
                                     dtype=da.float32)
        print(f"ayto {data.shape}, {data_array.shape}, {a.shape}, {normalized.shape}")
        return a