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
import numpy as np
import scipy.signal as signal
# noinspection PyProtectedMember
from traitlets import Int

from estimation_comparison.data_collection.estimator.base import EstimatorBase


class Autocorrelation(EstimatorBase):
    block_size = Int(1024)

    def estimate(self, data: np.ndarray) -> np.ndarray:
        def autocorrelate(block):
            mean = np.mean(block)
            var = np.var(block)
            if var == 0:
                # if the variance is zero it should have an autocorrelation of 1
                return np.ones(shape=(block.shape[0] * 2 - 1))
            zero_mean = np.subtract(block, mean)
            correlation = signal.correlate(zero_mean, zero_mean)
            return np.divide(correlation, (var * self.block_size))

        data_array = np.reshape(data[:len(data) // self.block_size * self.block_size], (-1, self.block_size))
        return np.apply_along_axis(autocorrelate, 1, data_array)
