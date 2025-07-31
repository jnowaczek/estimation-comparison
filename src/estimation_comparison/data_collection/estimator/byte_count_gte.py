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
import array
import itertools

import numpy as np
# noinspection PyProtectedMember
from traitlets import Int

from estimation_comparison.data_collection.estimator.base import EstimatorBase


class ByteCountGte(EstimatorBase):
    block_size = Int(None, allow_none=True)

    def estimate(self, data: np.ndarray) -> list[int]:
        if self.block_size:
            data = data.reshape((-1, self.block_size))
        else:
            data = data.reshape((1, -1))
        results = []

        for block in data:
            uniques, counts = np.unique(block, return_counts=True)

            # Use actual block size in case we get a small block at the end
            threshold = len(block) // 256
            results.append(len(np.where(counts >= threshold)[0]))

        return results
