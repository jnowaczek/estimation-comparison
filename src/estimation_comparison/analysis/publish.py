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
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import sklearn
from bokeh.models import Range1d
from bokeh.plotting import figure
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import permutation_test_score
from sklearn.pipeline import make_pipeline

from estimation_comparison.data_collection.estimator import Autocorrelation
from estimation_comparison.database import BenchmarkDatabase

db = BenchmarkDatabase(Path("benchmarks/benchmark.sqlite"))


@dataclass
class GraphSpec:
    title: str
    filename: str
    preprocessor_names: List[str]
    estimator_names: List[str]
    block_summary_func_names: List[str]
    file_summary_func_names: List[str]
    compressor_names: List[str]
    qualities: List[str]
    tags: List[str]


def render(graph: GraphSpec):
    fig = figure(x_range=Range1d(0, 100), width=800, height=600, x_axis_label="Percent Size Reduction",
                 y_axis_label="Estimator Metric")

    desc, rec = db.get_plot_dataframe(graph.preprocessor_names, graph.estimator_names, graph.compressor_names,
                                      graph.block_summary_func_names, graph.file_summary_func_names, graph.qualities,
                                      graph.tags)
    data = pd.DataFrame.from_records(rec, columns=[item[0] for item in desc])

    # for index, s in enumerate(series):
    #
    #     data["percent_size_reduction"] = (1.0 - (data["final_size"] / data["initial_size"])) * 100.0
    #     data.sort_values("percent_size_reduction", inplace=True)
    #     fig.scatter(x="percent_size_reduction", y="metric",
    #                 legend_label=f"{s.estimator} ({s.preprocessor}), {s.compressor}",
    #                 color=cc.b_glasbey_hv[index], source=data, alpha=0.2)
    #
    #     if s.lin_fit_show:
    #         linear = linear_fit(data["percent_size_reduction"], data["metric"])
    #         lin_conf = linear.eval_uncertainty(x=data["percent_size_reduction"], sigma=2)
    #         data["lin_fit"] = linear.best_fit
    #         data["lin_conf_lower"] = data["lin_fit"] - lin_conf
    #         data["lin_conf_upper"] = data["lin_fit"] + lin_conf
    #
    #     if s.quad_fit_show:
    #         quadratic = quadratic_fit(data["percent_size_reduction"], data["metric"])
    #         quad_conf = quadratic.eval_uncertainty(x=data["percent_size_reduction"], sigma=2)
    #         data["quad_fit"] = quadratic.best_fit
    #         data["quad_conf_lower"] = data["quad_fit"] - quad_conf
    #         data["quad_conf_upper"] = data["quad_fit"] + quad_conf
    #
    #     source = ColumnDataSource(data)
    #
    #     if s.lin_fit_show:
    #         fig.line(x="percent_size_reduction", y="lin_fit", source=source, color=cc.b_glasbey_hv[index])
    #         lin_band = Band(base="percent_size_reduction", lower="lin_conf_lower", upper="lin_conf_upper",
    #                         source=source, fill_color=cc.b_glasbey_hv[index], fill_alpha=0.5)
    #         fig.add_layout(lin_band)
    #
    #     if s.quad_fit_show:
    #         fig.line(x="percent_size_reduction", y="quad_fit", source=source, color=cc.b_glasbey_hv[index])
    #         quad_band = Band(base="percent_size_reduction", lower="quad_conf_lower", upper="quad_conf_upper",
    #                          source=source, fill_color=cc.b_glasbey_hv[index], fill_alpha=0.5)
    #         fig.add_layout(quad_band)

    return fig


def build_table(combinations: list[tuple[str, str, str]]):
    linear_results = pd.DataFrame(
        columns=["Compression Algorithm", "NRMSE", "p-value", "estimator", "summary statistic"])
    # quad_results = pd.DataFrame(columns=["Compression Algorithm", "NRMSE", "p-value", "estimator", "summary statistic"])

    compressor_names = [c[1] for c in db.get_compressors()]

    linear_pipeline = make_pipeline(LinearRegression())
    # quad_pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

    for (preprocessor_name, estimator_name, block_summary_func_name, file_summary_func_name) in combinations:
        for compressor_name in compressor_names:
            desc, rec = db.get_solo_plot_dataframe(preprocessor_name, estimator_name,
                                                   compressor_name,
                                                   block_summary_func_name,
                                                   file_summary_func_name)
            data = pd.DataFrame.from_records(rec, columns=[item[0] for item in desc])
            data["percent_size_reduction"] = (1.0 - (data["final_size"] / data["initial_size"])) * 100.0

            kfold = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=1337)
            scoring = ["neg_mean_squared_error"]
            linear_score, _, pvalue_linear = permutation_test_score(linear_pipeline, data[["metric"]],
                                                                    data["percent_size_reduction"],
                                                                    cv=kfold,
                                                                    scoring="neg_root_mean_squared_error",
                                                                    random_state=1337)
            # quad_score, _, pvalue_quad = permutation_test_score(quad_pipeline, data[["metric"]],
            #                                                     data["percent_size_reduction"],
            #                                                     cv=kfold,
            #                                                     scoring="neg_root_mean_squared_error",
            #                                                     random_state=1337)

            linear_results.loc[len(linear_results)] = [compressor_name, linear_score, pvalue_linear, estimator_name,
                                                       block_summary_func_name]
            # quad_results.loc[len(linear_results)] = [compressor_name, quad_score, pvalue_quad, estimator_name,
            #                                          block_summary_func_name]

    # SKL uses negative RMSE, fix that
    linear_results["RMSE"] = abs(linear_results["NRMSE"])
    # quad_results["RMSE"] = abs(quad_results["NRMSE"])

    return linear_results.sort_values("RMSE")  # , quad_results.sort_values("RMSE")


def autocorrelation_example():
    a = Autocorrelation(block_size=7)

    x = np.asarray([x / 10 for x in range(-3, 4)])
    y = a.estimate(x)[0]

    print(x, y)

    f = plt.figure()
    ax = f.add_subplot(1, 2, 1)
    ax.plot(x, x)
    ax = f.add_subplot(1, 2, 2)
    ax.plot(range(-6, 7), y)
    plt.show()


if __name__ == "__main__":
    # print("Bytecount")
    # bytecount_basic = filter(lambda c: c[0] == "entire_file" and c[1] == "bytecount_file", db.get_combinations())
    #
    # tables = build_table(list(bytecount_basic))
    # for table in tables:
    #     print(table[["Compression Algorithm", "RMSE", "p-value"]].to_latex(index=False))
    #
    # print("Entropy")
    # entropy_basic = filter(lambda c: c[0] == "entire_file" and c[1] == "entropy_bits", db.get_combinations())
    #
    # tables = build_table(list(entropy_basic))
    # for table in tables:
    #     print(table[["Compression Algorithm", "RMSE", "p-value"]].to_latex(index=False))

    print("Autocorrelation 972 notch")
    autocorrelation_972_basic = filter(
        lambda c: c[0] == "entire_file" and c[1] == "autocorrelation_972" and c[2] == "mean_inside_middle_notch_512",
        db.get_combinations())

    table = build_table(list(autocorrelation_972_basic))
    print(table[["Compression Algorithm", "summary statistic", "RMSE", "p-value"]].to_latex(index=False))
