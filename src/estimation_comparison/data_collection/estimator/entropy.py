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
import numpy as np
from scipy.stats import entropy
from traitlets import Int

from .base import EstimatorBase


class Entropy(EstimatorBase):
    base = Int(2)

    # I do kinda wish I could take credit for how simple this is, but...
    # https://stackoverflow.com/a/45091961
    def estimate(self, data: bytes) -> [int]:
        word, appearances = np.unique(np.frombuffer(data, dtype=np.dtype("B")), return_counts=True)
        return entropy(appearances, base=self.base)
