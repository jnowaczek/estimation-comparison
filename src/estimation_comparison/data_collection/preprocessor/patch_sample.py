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
import math
import random

import numpy as np
# noinspection PyProtectedMember
from traitlets import Int, Float

from estimation_comparison.data_collection.preprocessor import BaseSampler


class PatchSampler(BaseSampler[np.ndarray]):
    seed = Int(1337)
    # An (18, 18, 3) patch is 972 bytes
    patch_dim = Int(18)
    fraction = Float(0.1)

    def run(self, data: np.ndarray) -> np.ndarray:
        if self.patch_dim > data.shape[0]:
            raise ValueError(
                f"Requested patch height is too tall for supplied data: {self.patch_dim} > {data.shape[0]}")
        if self.patch_dim > data.shape[1]:
            raise ValueError(
                f"Requested patch width is too wide for supplied data: {self.patch_dim} > {data.shape[1]}")

        patch_start_rows = range(0, data.shape[0], self.patch_dim)
        patch_start_cols = range(0, data.shape[1], self.patch_dim)
        all_patches = list(itertools.product(patch_start_rows, patch_start_cols))
        random.seed(self.seed)
        sample_patches = random.sample(all_patches, math.floor(len(all_patches) * self.fraction))
        return np.hstack(
            [data[coord[0]:coord[0] + self.patch_dim, coord[1]:coord[1] + self.patch_dim, :].flatten() for coord in
             sample_patches])

# class PatchSamplerBytes(BaseSampler[bytes]):
#     seed = Int(1337)
#     patch_dim = Int(18)
#     chunk_size = Int(1024)
#     fraction = Float(0.1)
#
#     def run(self, data: bytes) -> bytes:
#         patches = image.extract_patches_2d(data,
#                                            patch_size=(self.patch_dim, self.patch_dim),
#                                            max_patches=self.fraction,
#                                            random_state=self.seed)
#         result = b""
#         for patch in patches:
#             result += patch.flatten()
#             result += b"\x00" * (self.chunk_size - len(result) % self.chunk_size)
#         return result
