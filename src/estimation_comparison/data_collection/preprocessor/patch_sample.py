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
import numpy as np
from sklearn.feature_extraction import image
from traitlets import Int, Float

from estimation_comparison.data_collection.preprocessor import BaseSampler


class PatchSampler(BaseSampler[np.ndarray]):
    seed = Int(1337)
    # An (18, 18, 3) patch is 972 bytes
    patch_dim = Int(18)
    fraction = Float(0.1)

    def run(self, data: np.ndarray) -> np.ndarray:
        # Too clever for me, I'll just import the thing
        return image.extract_patches_2d(data,
                                        patch_size=(self.patch_dim, self.patch_dim),
                                        max_patches=self.fraction,
                                        random_state=self.seed)
