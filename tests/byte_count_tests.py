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
import unittest

from estimation_comparison.algorithm.byte_count import ByteCount


class BasicByteCountTests(unittest.TestCase):
    basic = ByteCount({"block_size": 1024})

    def test_zeros(self):
        result = self.basic.estimate(b"0" * 1024)
        self.assertEqual(result, 1)

    def test_count(self):
        result = self.basic.estimate(bytes(range(256)) * 4)
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()