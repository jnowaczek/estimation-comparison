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
import math

import numpy as np


def max_outside_middle_notch(x, notch_width: int):
    try:
        return np.max(x[:math.floor((len(x) / 2) - notch_width)])
    except ValueError:
        return 0.0


def proportion_below_lag_cutoff(x, cutoff: int):
    return np.sum(np.abs(x[:cutoff])) / np.sum(np.abs(x))


def max_below_cutoff(x, cutoff: int):
    try:
        return np.max(x[:cutoff])
    except ValueError:
        return 0.0


def autocorrelation_lag(x, lag: int) -> float:
    try:
        return abs(x[len(x) // 2 + lag])
    except ValueError:
        return 0.0


def proportion_above_metric_cutoff(x, cutoff: float) -> float:
    return len(np.asarray(np.abs(x) > cutoff).nonzero()[0]) / len(x)
