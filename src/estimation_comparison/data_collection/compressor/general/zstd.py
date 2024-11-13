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
import zstandard
# noinspection PyProtectedMember
from traitlets import Int

from estimation_comparison.data_collection.compressor.general.base import GeneralCompressorBase


class ZstandardCompressor(GeneralCompressorBase):
    level = Int(3)

    def compress(self, data: bytes) -> bytes:
        ctx = zstandard.ZstdCompressor(level=self.level)
        return ctx.compress(data)
