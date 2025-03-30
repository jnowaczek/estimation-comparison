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
from pathlib import Path
from sklearn.linear_model import LinearRegression

import pandas as pd
import panel as pn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from wat import wat

from estimation_comparison.analysis.fit import linear_fit, quadratic_fit
from estimation_comparison.database import BenchmarkDatabase

db = BenchmarkDatabase(Path("benchmarks/benchmark.sqlite"))


def fit(preprocessor: str, estimator: str, compressor: str, block_summary_fn: str, file_summary_fn: str):
    desc, rec = db.get_solo_plot_dataframe(preprocessor, estimator, compressor, block_summary_fn,
                                           file_summary_fn)

    data = pd.DataFrame.from_records(rec, columns=[item[0] for item in desc])

    data["percent_size_reduction"] = (1.0 - (data["final_size"] / data["initial_size"])) * 100.0
    data.sort_values("percent_size_reduction", inplace=True)

    linear_model = linear_fit(data["percent_size_reduction"], data["metric"])
    linear_skl = LinearRegression().fit(data[["percent_size_reduction"]], data["metric"])
    print("linear", linear_model.best_values, (linear_skl.coef_, linear_skl.intercept_))
    quad_model = quadratic_fit(data["percent_size_reduction"], data["metric"])
    quad_skl = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(data[["percent_size_reduction"]], data["metric"])
    print("quad", quad_model.best_values, (quad_skl.steps[1][1].coef_, quad_skl.steps[1][1].intercept_))



template = pn.template.BootstrapTemplate(title="RAISE Explorer")

home_button = pn.widgets.Button(name="Index", button_type="primary")
home_button.js_on_click(code="window.location.href='/panel/'")
template.header.append(home_button)

fit("entire_file", "autocorrelation_972", "jxl_lossless",
    "proportion_above_metric_cutoff_0.2", "mean")

# kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=1337)
# for i, (train_index, test_index) in enumerate(kf.split(data)):
#     template.main.append([data.iloc[train_index.flatten()], data.iloc[test_index.flatten()]])

template.servable()
