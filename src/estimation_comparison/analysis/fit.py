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
from lmfit.models import ExponentialModel, LinearModel, QuadraticModel


def linear_fit(x, y):
    model = LinearModel()
    params = model.make_params(intercept=dict(value=0, vary=False))
    return model.fit(y, x=x, params=params)

def quadratic_fit(x, y):
    model = QuadraticModel()
    params = model.make_params(c=dict(value=0, vary=False))
    return model.fit(y, x=x, params=params)

def exponential_fit(x, y):
    model = ExponentialModel()
    return model.fit(y, x=x)
