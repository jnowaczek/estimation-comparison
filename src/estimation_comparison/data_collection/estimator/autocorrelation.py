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
            return signal.correlate(block, block)

        trimmed_data = data[:data.shape[0] // self.block_size * self.block_size]

        data_mean = np.mean(trimmed_data)
        data_std_dev_sqrd = np.std(trimmed_data) ** 2

        data_array = np.reshape(trimmed_data, (-1, self.block_size))

        data_array = np.subtract(data_array, data_mean)  # Zero mean
        correlation = np.apply_along_axis(autocorrelate, 1, data_array)
        return np.divide(correlation, data_std_dev_sqrd)
