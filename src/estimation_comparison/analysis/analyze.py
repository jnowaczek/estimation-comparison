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
import logging
import pickle
import webbrowser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import save

from estimation_comparison.analysis.plot import PlotHandler


class Analyze:
    def __init__(self, output_dir: str, workers: int | None, parallel=False):
        self.raw_data = None
        self.data = None
        self.output_dir = output_dir
        self.workers = workers
        self.plot_handler = None

        if parallel:
            self.process_pool = ProcessPoolExecutor(max_workers=workers)
        else:
            self.process_pool = None

    def _create_dataframe(self):
        """
        Create a pandas dataframe with the following structure:

        +-------+-------+-------+-----------+
        |       | Algo1 | Algo2 | Algo3     |
        +=======+=======+=======+===========+
        | File1 | 1     | [1]   | [[1],[1]] |
        +-------+-------+-------+-----------+
        | File2 | 2     | [2]   | [[2],[2]] |
        +-------+-------+-------+-----------+
        | File3 | 3     | [3]   | [[3],[3]] |
        +-------+-------+-------+-----------+

        :return:
        """
        self.data = pd.DataFrame.from_dict(self.raw_data, orient='index')
        logging.info(f"Loaded {self.data.shape} dataframe")

    def _create_computed_columns(self):
        def autocorrelation_scalar_max(data):
            peak = 0.0
            for chunk in data:
                for corr in chunk:
                    if abs(corr) > peak:
                        peak = abs(corr)
            return peak

        def autocorrelation_vector_max(data):
            peaks = []
            for chunk in data:
                peak = 0.0
                for corr in chunk:
                    if abs(corr) > peak:
                        peak = abs(corr)
                peaks.append(peak)
            return peaks

        def autocorrelation_mean(data):
            means = []
            for chunk in data:
                means.append(np.mean(chunk))
            return means

        # self.data["autocorrelation_1k_vector_max"] = self.data["autocorrelation_1k"].apply(autocorrelation_vector_max)
        # self.data["autocorrelation_1k_scalar_max"] = self.data["autocorrelation_1k"].apply(autocorrelation_scalar_max)
        # self.data["autocorrelation_1k_mean"] = self.data["autocorrelation_1k"].apply(autocorrelation_mean)
        # self.data["autocorrelation_1k_mean_mean"] = self.data["autocorrelation_1k_mean"].apply(np.mean)
        # self.data["autocorrelation_1k_mean_max"] = self.data["autocorrelation_1k_mean"].apply(max)

    def load(self, filename: Path):
        suffix = filename.suffix
        match suffix:
            case ".pkl":
                self.raw_data = pickle.load(open(filename, "rb"))
            case _:
                raise ValueError(f"Unsupported file type: {suffix}")
        logging.info(f"Loaded '{filename}'")
        self._create_dataframe()
        self._create_computed_columns()
        self.plot_handler = PlotHandler(self.data)

    def run(self):
        plots = {}

        for algorithm in self.data.columns:
            plots[algorithm] = self.plot_handler.individual_plot(algorithm)

        plots["compression_ratio_lzma"] = self.plot_handler.ratio_plot("lzma",
                                                                       algorithms=["entropy_bits",
                                                                                   "bytecount_file",
                                                                                   ]
                                                                       )

        plots["compression_ratio_gzip_max"] = self.plot_handler.ratio_plot("gzip_max",
                                                                           algorithms=["entropy_bits",
                                                                                       "bytecount_file"
                                                                                       ]
                                                                           )

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
        analyze.load(sorted(args.input_dir.glob("benchmark*"))[-1])

    analyze.run()
