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
import array
import itertools

# noinspection PyProtectedMember
from traitlets import Int

from estimation_comparison.data_collection.estimator.base import EstimatorBase


class ByteCount(EstimatorBase):
    block_size = Int(None, allow_none=True)

    def estimate(self, data: bytes) -> [int]:
        blocks = [data]
        results = []

        if self.block_size is not None:
            blocks = itertools.batched(data, self.block_size)

        for block in blocks:
            appearances = array.array("L", [0] * 256)

            for byte in block:
                appearances[byte] += 1

            # Use actual block size in case we get a small block at the end
            threshold = len(block) / len(appearances)
            results.append(len(list(filter(lambda x: x >= threshold, appearances.tolist()))))

        return results
