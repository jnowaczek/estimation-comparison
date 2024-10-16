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
from typing import Dict

import numpy as np
import scipy.signal as signal

from estimation_comparison.data_collection.estimator.estimator_base import EstimatorBase


class Autocorrelation(EstimatorBase):
    def __init__(self, parameters: Dict[str, any]):
        for parameter in ["block_size", "block_summary_function", "file_summary_function"]:
            if parameter not in parameters:
                raise ValueError(f"Missing required parameter: '{parameter}'")
        super().__init__(parameters)

    def estimate(self, data: bytes) -> any:
        acf = []
        for block in itertools.batched(data, self.parameters["block_size"]):
            mean = np.mean(block)
            normalized_block = np.subtract(block, mean)
            numerator = signal.correlate(normalized_block, normalized_block)
            denominator = np.sum(normalized_block * normalized_block)
            del normalized_block
            block_result = numerator / denominator
            del numerator, denominator
            if self.parameters["block_summary_function"] is not None:
                block_result = self.parameters["block_summary_function"](block_result)
            acf.append(block_result)
            del block_result

        if self.parameters["file_summary_function"] is not None:
            acf = self.parameters["file_summary_function"](acf)
        return acf
