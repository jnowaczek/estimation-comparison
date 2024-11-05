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
from random import random, seed
from typing import Iterable

import numpy as np
from traitlets import Bytes, Int, HasTraits, Float


class PatchSampler(HasTraits):
    seed = Bytes(b"deadbeef")
    patch_dim = Int(32)
    fraction = Float(0.1)

    def _make_patches(self, image: np.ndarray) -> Iterable[np.ndarray]:
        seed(self.seed)
        x, y = 0, 0

        while y < image.shape[1]:
            x = 0
            while x < image.shape[0]:
                if random() < self.fraction:
                    yield image[y:y + self.patch_dim, x:x + self.patch_dim, :]
                x += self.patch_dim
            y += self.patch_dim

    def run(self, image: np.ndarray) -> bytes:
        result = bytearray()
        for patch in self._make_patches(image):
            result.append(patch.flatten())
        return result
