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
import argparse
import itertools
import logging
import webbrowser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from bokeh.io import save
from bokeh.plotting import figure, show

from estimation_comparison.analysis.fit import exponential_fit, linear_fit, quadratic_fit
from estimation_comparison.analysis.plot import PlotHandler
from estimation_comparison.database import BenchmarkDatabase


class Analyze:
    def __init__(self, output_dir: str, workers: int | None, parallel=False):
        self.database: Optional[BenchmarkDatabase] = None
        self.data: pd.DataFrame | None = None
        self.output_dir = output_dir
        self.workers = workers
        self.plot_handler = None

        if parallel:
            self.process_pool = ProcessPoolExecutor(max_workers=workers)
        else:
            self.process_pool = None

    def _create_dataframes(self):
        # self.ratios = pd.DataFrame().from_records(self.database.get_all_ratios(),
        #                                           columns=["filename", "compressor", "ratio"])
        # self.metrics = pd.DataFrame().from_records(self.database.get_all_metric(),
        #                                            columns=["filename", "estimator", "metric"])
        description, records = self.database.get_dataframe()

        self.data = pd.DataFrame().from_records(records, columns=[item[0] for item in description])
        logging.info(f"Loaded {self.data.shape} dataframe")

    def load(self, filename: Path):
        suffix = filename.suffix
        match suffix:
            case ".pkl":
                raise ValueError(f"Pickle files not supported in this version: {suffix}")
            case ".sqlite":
                self.database = BenchmarkDatabase(filename)
            case _:
                raise ValueError(f"Unsupported file type: {suffix}")
        logging.info(f"Loaded '{filename}'")
        self._create_dataframes()
        self.plot_handler = PlotHandler(self.data)

    def run_explore(self):
        print(self.data.columns)
        print(self.data.dtypes)
        print(type(self.data["filename"][0]))
        # Passing opts to the explorer either takes forever with lots of graphs or doesn't work, set default instead
        # opts.defaults(opts.Scatter(hover_tooltips=["test"]))
        hve = self.data.hvplot.explorer()
        hve.show()

    def run_fit(self):

        est = self.data["estimator"].unique().tolist()
        pre = self.data["preprocessor"].unique().tolist()
        comp = self.data["compressor"].unique().tolist()

        for e in est:
            for (p, c) in itertools.product(pre, comp):
                subset = self.data.loc[(self.data["estimator"] == e) & (self.data["preprocessor"] == p)]
                subset = subset.sort_values("ratio")

                if not subset.empty:
                    x = subset.ratio.tolist()
                    y = subset.metric.tolist()

                    linear = linear_fit(x, y)
                    quadratic = quadratic_fit(x, y)
                    exponential = exponential_fit(x, y)

                    f = figure(y_range=(0, 1))
                    ly = linear.best_fit
                    qy = quadratic.best_fit
                    ey = exponential.best_fit

                    f.scatter(x=x, y=y)
                    f.line(x=x, y=ly, line_color="red", legend_label="Linear")
                    f.line(x=x, y=qy, line_color="orange", legend_label="Quadratic")
                    f.line(x=x, y=ey, line_color="purple", legend_label="Exponential")
                    show(f)

    def run(self):
        plots = {}

        self.data.hvplot(
            by=['estimator'],
            groupby=['preprocessor', 'compressor'],
            kind='scatter',
            x='ratio',
            y=['metric'],
            legend='bottom_right',
            widget_location='bottom',
        )

        # for compressor in self.data["compressor"].unique():
        #     for estimator in self.data["estimator"].unique():
        #         plots[(compressor, estimator)] = self.plot_handler.ratio_plot(compressor, estimator)

        # p = self.data.hvplot(
        #     by=['estimator'],
        #     groupby=['compressor'],
        #     height=1000,
        #     kind='scatter',
        #     logx=True,
        #     logy=True,
        #     x='ratio',
        #     y=['metric'],
        #     legend='bottom_right',
        #     widget_location='bottom',
        #     # hover_tooltips=[("Filename", "@filename")],
        #     use_index=True,
        # )
        # p.show()

        # for algorithm in filter(lambda x: "Parameters" not in x, self.data.columns):
        #     print(f"{algorithm}: {self.data[algorithm].corr(self.data['jxl'])}")

        save_time = datetime.now().isoformat(timespec="seconds")
        for name, p in plots.items():
            if p is not None:
                filename = f"{self.output_dir}/plot_{save_time}_{name}.html"
                save(p, filename, resources="cdn", title=name)
                if args.open_in_browser:
                    webbrowser.open_new_tab(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("-i", "--input_dir", type=Path, dest="input_dir", default="./benchmarks",
                        help="directory to load benchmark data from")
    parser.add_argument("-f", "--file", type=Path, help="path to input file to analyze")
    parser.add_argument("-p", "--parallel", dest="parallel", action="store_true",
                        help="enable parallelized plot generation")
    parser.add_argument("-w", "--workers", type=int, dest="workers", default=None,
                        help="number of parallel workers, defaults to number of cores")
    parser.add_argument("-o", "--output_dir", type=Path, dest="output_dir", default="./output",
                        help="plot output directory")
    parser.add_argument("-b", "--browser", dest="open_in_browser", action="store_true",
                        help="open plots in default browser after rendering")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    analyze = Analyze(args.output_dir, workers=args.workers, parallel=args.parallel)

    if args.file is not None:
        analyze.load(args.file)
    else:
        analyze.load(args.input_dir / "benchmark.sqlite")

    analyze.run_fit()
