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

from estimation_comparison.data_collection.estimator.entropy import Entropy


class BasicEntropyTests(unittest.TestCase):
    bits = Entropy({"base": 2})

    def test_zeros(self):
        result = self.bits.estimate(b"0" * 1024)
        self.assertEqual(0.0, result)

    def test_count(self):
        result = self.bits.estimate(bytes(range(256)) * 4)
        self.assertEqual(8, result)

    def test_text(self):
        result = self.bits.estimate(b"Standard English text usually falls somewhere between 3.5 and 5")
        self.assertAlmostEqual(4.228788210509104, result)

    def test_text_dog(self):
        result = self.bits.estimate(b"The quick brown fox jumped over the lazy dog")
        self.assertAlmostEqual(4.368522527728207, result)


if __name__ == '__main__':
    unittest.main()
