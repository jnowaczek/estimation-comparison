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
from imagecodecs import tiff_decode, jpeg_encode, tiff_check

from typing import Dict

from imagecodecs.imagecodecs import jpeg_encode
from traitlets import Bool

from estimation_comparison.data_collection.compressor.image.base import ImageCompressorBase


class JpegCompressor(ImageCompressorBase):
    lossless = Bool(False)

    def compress(self, data: bytes) -> bytes:
        if not tiff_check(data):
            raise ValueError("Input must be tiff")

        nda = tiff_decode(data)
        return jpeg_encode(nda)