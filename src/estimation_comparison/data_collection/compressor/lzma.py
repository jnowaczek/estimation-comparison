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
from typing import Dict
import lzma

from estimation_comparison.data_collection.compressor.compressor_base import CompressorBase


class LzmaCompressor(CompressorBase):
    def __init__(self, parameters: Dict[str, any]):
        for parameter in []:
            if parameter not in parameters:
                raise ValueError(f"Missing required parameter: '{parameter}'")
        super().__init__(parameters)

    def compress(self, data: bytes) -> bytes:
        return lzma.compress(data)
