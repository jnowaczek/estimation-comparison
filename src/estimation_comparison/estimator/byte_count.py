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
from typing import Dict

from estimation_comparison.estimator.estimator_base import EstimatorBase


class ByteCount(EstimatorBase):
    def __init__(self, parameters: Dict[str, any]):
        for parameter in ["block_size"]:
            if parameter not in parameters:
                raise ValueError(f"Missing required parameter: '{parameter}'")
        super().__init__(parameters)

    def estimate(self, data: bytes) -> int:
        appearances = array.array("H", [0] * 256)

        # TODO: Utilize block_size parameter, otherwise appearances can overflow
        for byte in data:
            appearances[byte] += 1

        threshold = self.parameters["block_size"] / len(appearances)
        return len(list(filter(lambda x: x > threshold, appearances.tolist())))
