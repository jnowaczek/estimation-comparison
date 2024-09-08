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
import abc
from typing import Dict

import numpy as np

from estimation_comparison.data_collection.algorithm_base import AlgorithmBase


class ImageCompressorBase:
    parameters: Dict[str, any]

    @abc.abstractmethod
    def __init__(self, parameters: Dict[str, any]):
        self.parameters = parameters

    @abc.abstractmethod
    def compress(self, data: np.ndarray) -> bytes:
        pass

    def ratio(self, data: np.ndarray) -> float:
        return len(data) / len(self.compress(data))

    def run(self, data: np.ndarray) -> float:
        return self.ratio(data)
