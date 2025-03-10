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

    def estimate(self, data: np.ndarray) -> any:
        def normalize(block):
            return np.subtract(block, np.mean(block))

        def autocorrelate(block):
            return signal.correlate(block, block)

        normalized = np.apply_along_axis(normalize, 1, data.reshape((self.block_size, -1)))
        numerators = np.apply_along_axis(autocorrelate, 1, normalized)
        denominators = np.apply_along_axis(autocorrelate, 1, normalized)
        block_result = np.divide(numerators, denominators, out=np.zeros_like(numerators), where=denominators != 0,
                                     dtype=np.float16)
        del numerators, denominators

        return block_result
