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
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer
from typing import List

import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from estimation_comparison.database import BenchmarkDatabase


@dataclass
class Fit:
    preprocessor_name: str
    estimator_name: str
    block_summary_func_name: str
    file_summary_func_name: str
    compressor_name: str
    scores: any

    def __lt__(self, other):
        return self.scores.mean() < other.scores.mean()


class Analyze:
    def __init__(self, input_dir: str, output_dir: str):
        self._init_time = default_timer()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.database = BenchmarkDatabase(Path(self.input_dir) / "benchmark.sqlite")

    def run(self):
        linear_results: List[Fit] = []
        quad_results = []

        combinations = self.database.get_combinations()
        compressor_names = [c[1] for c in self.database.get_compressors()]

        linear_pipeline = make_pipeline(LinearRegression())
        quad_pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

        for (preprocessor_name, estimator_name, block_summary_func_name, file_summary_func_name) in combinations:
            for compressor_name in compressor_names:
                desc, rec = self.database.get_solo_plot_dataframe(preprocessor_name, estimator_name, compressor_name,
                                                                  block_summary_func_name, file_summary_func_name)
                data = pd.DataFrame.from_records(rec, columns=[item[0] for item in desc])
                data["percent_size_reduction"] = (1.0 - (data["final_size"] / data["initial_size"])) * 100.0

                kfold = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=1337)
                linar_scores = cross_val_score(linear_pipeline, data[["metric"]], data["percent_size_reduction"], cv=kfold)
                quad_scores = cross_val_score(quad_pipeline, data[["metric"]], data["percent_size_reduction"], cv=kfold)
                linear_results.append(
                    Fit(preprocessor_name, estimator_name, block_summary_func_name, file_summary_func_name,
                        compressor_name, linar_scores))
                quad_results.append(
                    Fit(preprocessor_name, estimator_name, block_summary_func_name, file_summary_func_name,
                        compressor_name, quad_scores))

        print("=== Linear Fit ===")
        for x in (sorted(linear_results)):
            print(x)

        print("=== Quadratic Fit ===")
        for x in (sorted(quad_results)):
            print(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("-i", "--input_dir", type=Path, dest="input_dir", default="./benchmarks",
                        help="directory to load benchmark data from")
    parser.add_argument("-o", "--output_dir", type=Path, dest="output_dir", default="./analysis",
                        help="analysis output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    analyze = Analyze(args.input_dir, args.output_dir)

    analyze.run()
