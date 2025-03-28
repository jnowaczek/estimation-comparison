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
import itertools
import logging

import numpy as np
from bokeh.models import ColumnDataSource, FactorRange, HoverTool
from bokeh.palettes import Category20
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from pandas import DataFrame


class PlotHandler:
    def __init__(self, data: DataFrame):
        self._data = data
        self.palette = Category20[20]

    def individual_plot(self, algorithm: str) -> any:
        match algorithm:
            case "entropy_bits" | "bytecount_file" | "gzip_max" | "lzma":
                pass
            case "autocorrelation_1k" as a:
                return self.line_plot_all_files(a)
                pass
            case _:
                logging.error(f"Unable to plot unknown estimator: {algorithm}")
                return None

    def ratio_plot(self, x, y) -> any:
        df = self._data.loc[(self._data["compressor"] == x) & (self._data["estimator"] == y)]
        cds = ColumnDataSource(df)
        plot = figure(title=f"{x} Compression Ratio vs {y} Metric", sizing_mode="stretch_both",
                      tooltips=[("Estimator", "@estimator"), ("Metric", "$snap_y"), ("File name", "@filename")],
                      x_axis_type="log")
        plot.x_range.start = 0
        plot.scatter(
            x="ratio",
            y="metric",
            source=cds,
            legend_label=y
        )
        plot.xaxis.axis_label = f"{x} Compression Ratio"
        plot.yaxis.axis_label = f"{y} Metric"
        plot.legend.location = "top_left"
        plot.legend.click_policy = "mute"
        return plot

    def line_plot_all_files(self, algorithm: str) -> any:
        plot = figure(title=f"{algorithm} vs Block Index", sizing_mode="stretch_both",
                      tooltips=[("File name", "$name"), ("Metric", "$snap_y")])
        for ((file, data), color) in zip(self._data[algorithm].items(), itertools.cycle(self.palette)):
            plot.scatter(
                list(range(0, len(data[0]))),
                data[0],
                name=file,
                color=color,
            )
        plot.add_tools(HoverTool(tooltips=None))
        return plot

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
        plot.xaxis.major_label_text_font_size = "1em"
        plot.xaxis.group_text_font_size = "1em"
        return plot
