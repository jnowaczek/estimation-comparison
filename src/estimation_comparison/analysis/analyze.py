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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from estimation_comparison.analysis.plot import plot_scalar, plot_vector


class Analyze:
    def __init__(self, output_dir: str, workers: int | None, parallel=False):
        self.data = None
        self.output_dir = output_dir
        self.workers = workers

        if parallel:
            self.process_pool = ProcessPoolExecutor(max_workers=workers)
        else:
            self.process_pool = None

    def load(self, filename: str):
        suffix = Path(filename).suffix
        match suffix:
            case ".feather":
                self.data = pd.read_feather(filename)
            case _:
                raise ValueError(f"Unsupported file type: {suffix}")

    def run(self):
        for estimator in self.data.keys():
            if estimator == "entropy_bits":
                plot_scalar(self.data["entropy_bits"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="input file to analyze")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("-p", "--parallel", dest="parallel", action="store_true",
                        help="enable parallelized plot generation")
    parser.add_argument("-w", "--workers", type=int, dest="workers", default=None,
                        help="number of parallel workers, defaults to number of cores")
    parser.add_argument("-o", "--output_dir", type=str, dest="output_dir", default=".",
                        help="plot output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    analyze = Analyze(args.output_dir, workers=args.workers, parallel=args.parallel)
    analyze.load(args.file)

    analyze.run()
