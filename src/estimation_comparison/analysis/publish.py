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
        columns=["Compression Algorithm", "NRMSE", "p-value", "estimator", "summary statistic", "preprocessor"])
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
                                                       block_summary_func_name, preprocessor_name]
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


def build_basic_ac_table():
    print("Autocorrelation 972 notch")
    autocorrelation_972_basic = filter(
        lambda c: c[0] == "entire_file" and c[1] == "autocorrelation_972",
        db.get_combinations())

    table = build_table(list(autocorrelation_972_basic))
    table.to_csv("~/ac972.csv")
    return table


def heatmap_helper(grouped, grouped_p, name, size, ylabels=None):
    fig, ax = plt.subplots(figsize=size, dpi=300, layout="constrained")
    im = ax.imshow(grouped, cmap="Greys", vmin=0, vmax=15)
    ax.set_xticks(range(len(grouped.columns)), grouped.columns, rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(grouped.index)), grouped.index if ylabels is None else ylabels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.spines[:].set_color("white")
    ax.set_xticks(np.arange(len(grouped.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(grouped.index) + 1) - 0.5, minor=True)
    ax.tick_params(which="both", bottom=False, left=False, top=False, right=False)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    for i in range(len(grouped.columns)):
        for j in range(len(grouped.index)):
            pval = f"P={grouped_p.iloc[j, i]:.2f}" if grouped_p.iloc[j, i] > 0.01 else "P<0.01"
            text = ax.text(i, j, f"{round(grouped.iloc[j, i], 2)}\n{pval}", ha="center", va="center",
                           size="xx-small",
                           color="black" if grouped.iloc[j, i] < 7.5 else "white")
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.01, )
    cbar.ax.set_yticks(np.arange(0, 15.01, 5))
    cbar.ax.set_ylabel("Model RMSE (Percent Size Reduction)")
    fig.savefig(f"plots/{name}.png")
    plt.close(fig)


def bsf_error_heatmap(df: pd.DataFrame, bsfs: list[str], name: str, size: tuple[float, float], ylabels=None):
    grouped = df.pivot(index="summary statistic", columns="Compression Algorithm", values="RMSE")
    grouped = grouped[grouped.index.isin(bsfs)]
    grouped_p = df.pivot(index="summary statistic", columns="Compression Algorithm", values="p-value")
    grouped_p = grouped_p[grouped_p.index.isin(bsfs)]
    return heatmap_helper(grouped, grouped_p, name, size, ylabels=ylabels)


def estimator_error_heatmap(df: pd.DataFrame, estimators: list[str], name: str, size: tuple[float, float],
                            ylabels=None):
    grouped = df.pivot(index="estimator", columns="Compression Algorithm", values="RMSE")
    grouped = grouped[grouped.index.isin(estimators)]
    grouped_p = df.pivot(index="estimator", columns="Compression Algorithm", values="p-value")
    grouped_p = grouped_p[grouped_p.index.isin(estimators)]
    return heatmap_helper(grouped, grouped_p, name, size, ylabels=ylabels)


def sampled_error_heatmap(df: pd.DataFrame, preprocessors: list[str], name: str, size: tuple[float, float],
                          ylabels=None):
    grouped = df.pivot(index="preprocessor", columns="Compression Algorithm", values="RMSE")
    grouped = grouped[grouped.index.isin(preprocessors)]
    grouped_p = df.pivot(index="preprocessor", columns="Compression Algorithm", values="p-value")
    grouped_p = grouped_p[grouped_p.index.isin(preprocessors)]
    return heatmap_helper(grouped, grouped_p, name, size, ylabels=ylabels)


if __name__ == "__main__":
    # autocorrelation_basic_df = build_basic_ac_table()
    #
    # # lag
    # bsf_error_heatmap(autocorrelation_basic_df, ["lag_0", "lag_1", "lag_3"], "basic/lag", (6, 3),
    #                   ylabels=["0", "1", "3"])
    #
    # # metric_cutoff
    # bsf_error_heatmap(autocorrelation_basic_df, ["proportion_above_metric_cutoff_0.05",
    #                                              "proportion_above_metric_cutoff_0.1",
    #                                              "proportion_above_metric_cutoff_0.15",
    #                                              "proportion_above_metric_cutoff_0.2",
    #                                              "proportion_above_metric_cutoff_0.25",
    #                                              "proportion_above_metric_cutoff_0.3",
    #                                              "proportion_above_metric_cutoff_0.35",
    #                                              "proportion_above_metric_cutoff_0.4",
    #                                              "proportion_above_metric_cutoff_0.45",
    #                                              "proportion_above_metric_cutoff_0.5",
    #                                              "proportion_above_metric_cutoff_0.55",
    #                                              "proportion_above_metric_cutoff_0.6",
    #                                              "proportion_above_metric_cutoff_0.65",
    #                                              "proportion_above_metric_cutoff_0.7",
    #                                              "proportion_above_metric_cutoff_0.75",
    #                                              "proportion_above_metric_cutoff_0.8",
    #                                              "proportion_above_metric_cutoff_0.85",
    #                                              "proportion_above_metric_cutoff_0.9",
    #                                              "proportion_above_metric_cutoff_0.95"]
    #                   , "basic/metric_cutoff", (6, 7), ylabels=[f"{x * 0.05:.2f}" for x in range(1, 20)])
    #
    # # Mean inside
    # bsf_error_heatmap(autocorrelation_basic_df, ["mean_inside_middle_notch_64",
    #                                              "mean_inside_middle_notch_128",
    #                                              "mean_inside_middle_notch_256",
    #                                              "mean_inside_middle_notch_512"], "basic/mean_inside", (6, 3),
    #                   ylabels=["64", "128", "256", "512"])
    #
    # # Max outside
    # bsf_error_heatmap(autocorrelation_basic_df, ["max_outside_middle_notch_64"], "basic/max_outside", (6, 3),
    #                   ylabels=["64"])

    # autocorrelation_linear = filter(
    #     lambda c: (c[0] == "linear_random_25%" or c[0] == "linear_random_50%" or c[0] == "linear_random_75%") and c[
    #         1] == "autocorrelation_972",
    #     db.get_combinations())
    #
    # autocorrelation_linear_table = build_table(list(autocorrelation_linear))
    # autocorrelation_linear_table.to_csv("~/ac972_linear.csv")
    #
    # autocorrelation_patch = filter(
    #     lambda c: (c[0] == "patch_random_25%" or c[0] == "patch_random_50%" or c[0] == "patch_random_75%") and c[
    #         1] == "autocorrelation_972",
    #     db.get_combinations())
    #
    # autocorrelation_patch_table = build_table(list(autocorrelation_patch))
    # autocorrelation_patch_table.to_csv("~/ac972_patch.csv")

    autocorrelation_linear_table = pd.read_csv("~/ac972_linear.csv")
    autocorrelation_patch_table = pd.read_csv("~/ac972_patch.csv")

    # lag
    for x in ["25", "50", "75"]:
        bsf_error_heatmap(
            autocorrelation_linear_table[autocorrelation_linear_table["preprocessor"] == f"linear_random_{x}%"],
            ["lag_0", "lag_1", "lag_3"], f"linear/lag_{x}", (6, 3),
            ylabels=["0", "1", "3"])

        # metric_cutoff
        bsf_error_heatmap(
            autocorrelation_linear_table[autocorrelation_linear_table["preprocessor"] == f"linear_random_{x}%"],
            ["proportion_above_metric_cutoff_0.05",
             "proportion_above_metric_cutoff_0.1",
             "proportion_above_metric_cutoff_0.15",
             "proportion_above_metric_cutoff_0.2",
             "proportion_above_metric_cutoff_0.25",
             "proportion_above_metric_cutoff_0.3",
             "proportion_above_metric_cutoff_0.35",
             "proportion_above_metric_cutoff_0.4",
             "proportion_above_metric_cutoff_0.45",
             "proportion_above_metric_cutoff_0.5",
             "proportion_above_metric_cutoff_0.55",
             "proportion_above_metric_cutoff_0.6",
             "proportion_above_metric_cutoff_0.65",
             "proportion_above_metric_cutoff_0.7",
             "proportion_above_metric_cutoff_0.75",
             "proportion_above_metric_cutoff_0.8",
             "proportion_above_metric_cutoff_0.85",
             "proportion_above_metric_cutoff_0.9",
             "proportion_above_metric_cutoff_0.95"]
            , f"linear/metric_cutoff_{x}", (6, 7), ylabels=[f"{x * 0.05:.2f}" for x in range(1, 20)])

        # Mean inside
        bsf_error_heatmap(
            autocorrelation_linear_table[autocorrelation_linear_table["preprocessor"] == f"linear_random_{x}%"],
            ["mean_inside_middle_notch_64",
             "mean_inside_middle_notch_128",
             "mean_inside_middle_notch_256",
             "mean_inside_middle_notch_512"], f"linear/mean_inside_{x}", (6, 3),
            ylabels=["64", "128", "256", "512"])

        # Max outside
        bsf_error_heatmap(
            autocorrelation_linear_table[autocorrelation_linear_table["preprocessor"] == f"linear_random_{x}%"],
            ["max_outside_middle_notch_64"], f"linear/max_outside_{x}", (6, 3),
            ylabels=["64"])

        # lag
        bsf_error_heatmap(
            autocorrelation_patch_table[autocorrelation_patch_table["preprocessor"] == f"patch_random_{x}%"],
            ["lag_0", "lag_1", "lag_3"], f"patch/lag_{x}", (6, 3),
            ylabels=["0", "1", "3"])

        # metric_cutoff
        bsf_error_heatmap(
            autocorrelation_patch_table[autocorrelation_patch_table["preprocessor"] == f"patch_random_{x}%"],
            ["proportion_above_metric_cutoff_0.05",
             "proportion_above_metric_cutoff_0.1",
             "proportion_above_metric_cutoff_0.15",
             "proportion_above_metric_cutoff_0.2",
             "proportion_above_metric_cutoff_0.25",
             "proportion_above_metric_cutoff_0.3",
             "proportion_above_metric_cutoff_0.35",
             "proportion_above_metric_cutoff_0.4",
             "proportion_above_metric_cutoff_0.45",
             "proportion_above_metric_cutoff_0.5",
             "proportion_above_metric_cutoff_0.55",
             "proportion_above_metric_cutoff_0.6",
             "proportion_above_metric_cutoff_0.65",
             "proportion_above_metric_cutoff_0.7",
             "proportion_above_metric_cutoff_0.75",
             "proportion_above_metric_cutoff_0.8",
             "proportion_above_metric_cutoff_0.85",
             "proportion_above_metric_cutoff_0.9",
             "proportion_above_metric_cutoff_0.95"]
            , f"patch/metric_cutoff_{x}", (6, 7), ylabels=[f"{x * 0.05:.2f}" for x in range(1, 20)])

        # Mean inside
        bsf_error_heatmap(
            autocorrelation_patch_table[autocorrelation_patch_table["preprocessor"] == f"patch_random_{x}%"],
            ["mean_inside_middle_notch_64",
             "mean_inside_middle_notch_128",
             "mean_inside_middle_notch_256",
             "mean_inside_middle_notch_512"], f"patch/mean_inside_{x}", (6, 3),
            ylabels=["64", "128", "256", "512"])

        # Max outside
        bsf_error_heatmap(
            autocorrelation_patch_table[autocorrelation_patch_table["preprocessor"] == f"patch_random_{x}%"],
            ["max_outside_middle_notch_64"], f"patch/max_outside_{x}", (6, 3),
            ylabels=["64"])

    # bytecount = filter(
    #     lambda c: c[0] == "entire_file" and c[1] == "bytecount_file", db.get_combinations())
    # entropy = filter(
    #     lambda c: c[0] == "entire_file" and c[1] == "entropy_bits", db.get_combinations())
    #
    # bytecount_table = build_table(list(bytecount))
    # entropy_table = build_table(list(entropy))
    # estimator_error_heatmap(bytecount_table, ["bytecount_file"], "basic/bytecount", (6, 3))
    # estimator_error_heatmap(entropy_table, ["entropy_bits"], "basic/entropy", (6, 3))
    #
    # bytecount_linear = filter(
    #     lambda c: (c[0] == "linear_random_25%" or c[0] == "linear_random_50%" or c[0] == "linear_random_75%") and c[
    #         1] == "bytecount_file", db.get_combinations())
    #
    # bytecount_linear_table = build_table(list(bytecount_linear))
    # sampled_error_heatmap(bytecount_linear_table, ["linear_random_25%", "linear_random_50%", "linear_random_75%"],
    #                       "linear/bytecount", (6, 3))
    #
    # entropy_linear = filter(
    #     lambda c: (c[0] == "linear_random_25%" or c[0] == "linear_random_50%" or c[0] == "linear_random_75%") and c[
    #         1] == "entropy_bits", db.get_combinations())
    #
    # entropy_linear_table = build_table(list(entropy_linear))
    # sampled_error_heatmap(entropy_linear_table, ["linear_random_25%", "linear_random_50%", "linear_random_75%"],
    #                       "linear/entropy", (6, 3))
    #
    # bytecount_patch = filter(
    #     lambda c: (c[0] == "patch_random_25%" or c[0] == "patch_random_50%" or c[0] == "patch_random_75%") and c[
    #         1] == "bytecount_file", db.get_combinations())
    # bytecount_patch_table = build_table(list(bytecount_patch))
    # sampled_error_heatmap(bytecount_patch_table, ["patch_random_25%", "patch_random_50%", "patch_random_75%"],
    #                       "patch/bytecount", (6, 3))
    #
    # entropy_patch = filter(
    #     lambda c: (c[0] == "patch_random_25%" or c[0] == "patch_random_50%" or c[0] == "patch_random_75%") and c[
    #         1] == "entropy_bits", db.get_combinations())
    #
    # entropy_patch_table = build_table(list(entropy_patch))
    # sampled_error_heatmap(entropy_patch_table, ["patch_random_25%", "patch_random_50%", "patch_random_75%"],
    #                       "patch/entropy", (6, 3))
