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
import math
import random

import numpy as np
# noinspection PyProtectedMember
from traitlets import Int, Float

from estimation_comparison.data_collection.preprocessor import BaseSampler


class LinearSampler(BaseSampler[np.ndarray]):
    seed = Int(1337)
    # An (18, 18, 3) patch is 972 bytes
    patch_len = Int(972)
    fraction = Float(0.1)

    def run(self, data: np.ndarray) -> np.ndarray:
        data.reshape((-1, 1, data.shape[-1]))
        if self.patch_len > data.shape[0]:
            raise ValueError(
                f"Requested patch length is too long for supplied data: {self.patch_len} > {data.shape[0]}")

        patch_start_indexes = list(filter(lambda x: x + self.patch_len < data.shape[0],
                                          range(0, data.shape[0], self.patch_len)))
        random.seed(self.seed)
        sample_patches = random.sample(patch_start_indexes,
                                       max(math.floor(len(patch_start_indexes) * self.fraction), 1))
        return np.hstack([data[coord:coord + self.patch_len, :] for coord in sample_patches])
