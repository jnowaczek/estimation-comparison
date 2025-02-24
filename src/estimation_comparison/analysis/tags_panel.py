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

import colorcet as cc
import pandas as pd
import panel as pn
from bokeh.models import Band, ColumnDataSource
from bokeh.plotting import figure

from estimation_comparison.analysis.fit import quadratic_fit, linear_fit
from estimation_comparison.database import BenchmarkDatabase

db = BenchmarkDatabase(Path("benchmarks/benchmark.sqlite"))

tag_name_to_id: {str, int} = {row[1]: row[0] for row in db.get_tags()}
# est_list: [Estimator] = db.get_estimators()

pre_select = pn.widgets.Select(name="Preprocessor", options=[entry[1] for entry in db.get_preprocessors()])
est_select = pn.widgets.Select(name="Estimator", options=[entry[1] for entry in db.get_estimators()])
comp_select = pn.widgets.Select(name="Compressor", options=[entry[1] for entry in db.get_compressors()])
tag_select = pn.widgets.MultiChoice(name="RAISE Tag", value=list(tag_name_to_id.keys()),
                                    options=list(tag_name_to_id.keys()))

lin_fit_show = pn.widgets.Checkbox(name="Show linear fit", value=False)
quad_fit_show = pn.widgets.Checkbox(name="Show quadratic fit", value=False)

template = pn.template.BootstrapTemplate(
    title="RAISE Tags Explorer",
    sidebar=[pre_select, est_select, comp_select, tag_select, lin_fit_show, quad_fit_show])


def plot(preprocessor: str, estimator: str, compressor: str, tags: [str],
         show_linear: bool = False, show_quadratic: bool = False):
    data_by_tag_name = dict()
    for tag_name in tags:
        desc, rec = db.get_solo_tag_plot_dataframe(preprocessor, estimator, compressor, tag_name)
        data_by_tag_name[tag_name] = pd.DataFrame.from_records(rec, columns=[item[0] for item in desc])

    fig = figure(y_range=(0, 10), x_range=(0, 100))
    fig.sizing_mode = "scale_both"

    if not data_by_tag_name.keys():
        fig.scatter()

    for tag_name, data in data_by_tag_name.items():
        tag_color = cc.glasbey_dark[tag_name_to_id[tag_name]]

        data["percent_size_reduction"] = (1.0 - (data["final_size"] / data["initial_size"])) * 100.0
        data.sort_values("percent_size_reduction", inplace=True)
        fig.scatter(x="percent_size_reduction", y="metric", source=data, alpha=0.2, color=tag_color)

        if show_quadratic:
            quadratic = quadratic_fit(data["percent_size_reduction"], data["metric"])
            quad_conf = quadratic.eval_uncertainty(x=data["percent_size_reduction"], sigma=2)
            data["quad_fit"] = quadratic.best_fit
            data["quad_conf_lower"] = data["quad_fit"] - quad_conf
            data["quad_conf_upper"] = data["quad_fit"] + quad_conf

        if show_linear:
            linear = linear_fit(data["percent_size_reduction"], data["metric"])
            lin_conf = linear.eval_uncertainty(x=data["percent_size_reduction"], sigma=2)
            data["lin_fit"] = linear.best_fit
            data["lin_conf_lower"] = data["lin_fit"] - lin_conf
            data["lin_conf_upper"] = data["lin_fit"] + lin_conf

        source = ColumnDataSource(data)

        if show_linear:
            fig.line(x="percent_size_reduction", y="lin_fit", color=tag_color,
                     legend_label=f"{tag_name} Linear fit", source=source)
            lin_band = Band(base="percent_size_reduction", lower="lin_conf_lower", upper="lin_conf_upper",
                             source=source, fill_color=tag_color, fill_alpha=0.5)
            fig.add_layout(lin_band)

        if show_quadratic:
            fig.line(x="percent_size_reduction", y="quad_fit", color=tag_color,
                     legend_label=f"{tag_name} Quadratic fit", source=source)
            quad_band = Band(base="percent_size_reduction", lower="quad_conf_lower", upper="quad_conf_upper",
                             source=source, fill_color=tag_color, fill_alpha=0.5)
            fig.add_layout(quad_band)

    return pn.pane.Bokeh(fig)

home_button = pn.widgets.Button(name="Index", button_type="primary")
home_button.js_on_click(code="window.location.href='/'")
template.header.append(home_button)


bokeh_pane = pn.bind(plot, pre_select, est_select, comp_select, tag_select, lin_fit_show, quad_fit_show)

template.main.append(bokeh_pane)

# table = pn.widgets.Tabulator(data)
# template.main.append(table)

template.servable()
