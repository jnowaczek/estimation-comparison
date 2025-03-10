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
import unittest

import numpy as np

from estimation_comparison.data_collection.estimator.byte_count import ByteCount


class BasicByteCountTests(unittest.TestCase):
    kilobyte = ByteCount(block_size=1024)
    half_kilobyte = ByteCount(block_size=512)
    quarter_kilobyte = ByteCount(block_size=256)

    def test_zeros_kilo(self):
        result = self.kilobyte.estimate(np.frombuffer(b"0" * 1024, dtype=np.uint8))
        self.assertEqual([1], result)

    def test_count_kilo(self):
        data = bytes(range(256)) * 4
        result = self.kilobyte.estimate(np.frombuffer(data, dtype=np.uint8))
        self.assertEqual([0], result)

    def test_zeros_half(self):
        result = self.half_kilobyte.estimate(np.frombuffer(b"0" * 1024, dtype=np.uint8))
        self.assertEqual([1, 1], result)

    def test_count_half(self):
        result = self.half_kilobyte.estimate(np.frombuffer(bytes(range(256)) * 4, dtype=np.uint8))
        self.assertEqual([0, 0], result)

    def test_zeros_quarter(self):
        result = self.quarter_kilobyte.estimate(np.frombuffer(b"0" * 1024, dtype=np.uint8))
        self.assertEqual([1, 1, 1, 1], result)

    def test_count_quarter(self):
        result = self.quarter_kilobyte.estimate(np.frombuffer(bytes(range(256)) * 4, dtype=np.uint8))
        self.assertEqual([0, 0, 0, 0], result)


class SingleBlockByteCountTests(unittest.TestCase):
    single_block = ByteCount(block_size=None)

    def test_zeros_5k(self):
        result = self.single_block.estimate(np.frombuffer(b"0" * 1024 * 5, dtype=np.uint8))
        self.assertEqual([1], result)

    def test_two_numbers_5k(self):
        result = self.single_block.estimate(np.frombuffer((b"0" * 512 + b"1" * 512) * 5, dtype=np.uint8))
        self.assertEqual([2], result)

    def test_count_5k(self):
        data = np.frombuffer(bytes(range(256)) * 4 * 5, dtype=np.uint8)
        result = self.single_block.estimate(data)
        self.assertEqual([0], result)


if __name__ == '__main__':
    unittest.main()
