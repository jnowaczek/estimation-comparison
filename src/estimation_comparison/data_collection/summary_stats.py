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
import math

import numpy as np
import dask.array as da


def max_outside_middle_notch(x: da.Array, notch_width: int):
    try:
        return da.max(x[:math.floor((x.shape[0] / 2) - notch_width)])
    except ValueError:
        return 0.0


def proportion_below_lag_cutoff(x: da.Array, cutoff: int):
    return da.sum(da.abs(x[:cutoff])) / da.sum(da.abs(x)).compute()


def max_below_cutoff(x: da.Array, cutoff: int):
    try:
        return da.max(x[:cutoff]).compute()
    except ValueError:
        return 0.0


def autocorrelation_lag(x: da.Array, lag: int) -> float:
    try:
        print(x.shape)
        # return da.abs(x[x.shape[0] // 2 + lag]).compute()
    except ValueError:
        return 0.0


def proportion_above_metric_cutoff(x: da.Array, cutoff: float) -> float:
    return len(da.asarray(da.abs(x) > cutoff).nonzero()[0]) / x.shape[0].compute()
