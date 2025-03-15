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
from abc import ABC

import numpy as np
# noinspection PyProtectedMember
from traitlets import HasTraits


class BaseSamplerMeta(type(ABC), type(HasTraits)):
    pass

class BaseSampler[T](ABC, HasTraits, metaclass=BaseSamplerMeta):
    def run(self, data: T) -> T:
        pass

class FlattenSampler(BaseSampler[np.ndarray]):
    def run(self, data: np.ndarray) -> np.ndarray:
        return data.flatten()
