#  Copyright (C) 2024 Julian Nowaczek.
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
import scipy.signal as signal
# noinspection PyProtectedMember
from traitlets import Callable, Int

from estimation_comparison.data_collection.estimator.base import EstimatorBase


class Autocorrelation(EstimatorBase):
    block_size = Int(1024)
    block_summary_fn = Callable()
    file_summary_fn = Callable()

    def estimate(self, data: np.ndarray) -> any:
        acf = []
        flat = data.tobytes()
        del data
        for block in itertools.batched(flat, self.block_size):
            mean = np.mean(block)
            normalized_block = np.subtract(block, mean)
            numerator = signal.correlate(normalized_block, normalized_block)
            denominator = np.sum(normalized_block * normalized_block)
            del normalized_block
            block_result = numerator / denominator
            del numerator, denominator
            block_result = self.block_summary_fn(block_result)
            acf.append(block_result)
            del block_result

        acf = self.file_summary_fn(acf)
        return acf
