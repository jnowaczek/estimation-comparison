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
import functools
import logging
from typing import Callable

import numpy as np
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, show
from bokeh.transform import factor_cmap
from pandas import DataFrame


class PlotHandler:
    def __init__(self, data):
        self._data = data
        self.palette = Category10[10]

    def individual_plot(self, algorithm: str) -> any:
        match algorithm:
            case "entropy_bits" | "bytecount_file" | "gzip_max" | "lzma":
                pass
            case _:
                logging.error(f"Unable to plot unknown estimator: {algorithm}")
                return None

    def ratio_plot(self) -> any:
        return self.adjacent_bars_all_files("Compression Ratio", ["gzip_max", "lzma"])

    def adjacent_bars_all_files(self, title: str, algorithms: [str]) -> any:
        files = self._data[algorithms[0]].keys()
        x = [(file, algorithm) for file in files for algorithm in algorithms]
        data = sum(list(tuple(self._data[d][f] for d in algorithms) for f in files), ())
        source = ColumnDataSource(data=dict(x=x, metric=data))

        plot = figure(title=title, x_range=FactorRange(*x), sizing_mode="stretch_width")
        plot.vbar(x="x",
                  top="metric",
                  source=source,
                  fill_color=factor_cmap("x", palette=self.palette, factors=algorithms, start=1, end=2))
        plot.xaxis.major_label_orientation = np.pi / 2
        plot.xaxis.group_label_orientation = np.pi / 2
        return plot
