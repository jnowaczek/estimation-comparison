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
import param
from bokeh.models import ColumnDataSource, Band
from bokeh.plotting import figure
from param import Parameterized

from estimation_comparison.analysis.fit import quadratic_fit, linear_fit
from estimation_comparison.database import BenchmarkDatabase

db = BenchmarkDatabase(Path("benchmarks/benchmark.sqlite"))


class Series(Parameterized):
    preprocessor = param.String()
    estimator = param.String()
    compressor = param.String()
    lin_fit_show = param.Boolean()
    quad_fit_show = param.Boolean()


class SeriesPlots(pn.viewable.Viewer):
    value = param.List(item_type=Series)

    def __init__(self, **params):
        super().__init__(**params)

    def add_series(self, series):
        self.value = [*self.value, series]


class SeriesInput(pn.viewable.Viewer):
    value: Series = param.ClassSelector(class_=Series)

    def __panel__(self):
        pre_select = pn.widgets.Select(name="Preprocessor", options=[entry[1] for entry in db.get_preprocessors()])
        est_select = pn.widgets.Select(name="Estimator", options=[entry[1] for entry in db.get_estimators()])
        comp_select = pn.widgets.Select(name="Compressor", options=[entry[1] for entry in db.get_compressors()])

        lin_fit_show = pn.widgets.Checkbox(name="Show linear fit", value=False)
        quad_fit_show = pn.widgets.Checkbox(name="Show quadratic fit", value=False)

        add_button = pn.widgets.Button(name="Add", button_type="primary", align="center")
        plot_controls = pn.layout.WidgetBox("## Configure series", pre_select, est_select, comp_select, lin_fit_show,
                                            quad_fit_show,
                                            add_button)

        @pn.depends(pre_select, est_select, comp_select, lin_fit_show, quad_fit_show, add_button, watch=True)
        def create_series(pre_select, est_select, comp_select, lin_fit_show, quad_fit_show, add_button):
            if add_button:
                self.value = Series(preprocessor=pre_select, estimator=est_select,
                                    compressor=comp_select, lin_fit_show=lin_fit_show,
                                    quad_fit_show=quad_fit_show)

        return plot_controls


class PlotEditor(pn.viewable.Viewer):
    value: SeriesPlots = param.ClassSelector(class_=SeriesPlots)

    @param.depends("value.value")
    def _layout(self):
        series = self.value.value
        return self._plot(series)

    @staticmethod
    def _plot(series):
        fig = figure(x_range=(0, 100))
        fig.sizing_mode = "scale_both"

        for index, s in enumerate(series):
            desc, rec = db.get_solo_tag_plot_dataframe(s.preprocessor, s.estimator, s.compressor, "outdoor")

            data = pd.DataFrame.from_records(rec, columns=[item[0] for item in desc])

            data["percent_size_reduction"] = (1.0 - (data["final_size"] / data["initial_size"])) * 100.0
            data.sort_values("percent_size_reduction", inplace=True)
            fig.scatter(x="percent_size_reduction", y="metric", color=cc.b_glasbey_hv[index], source=data, alpha=0.2)

            if s.lin_fit_show:
                linear = linear_fit(data["percent_size_reduction"], data["metric"])
                lin_conf = linear.eval_uncertainty(x=data["percent_size_reduction"], sigma=2)
                data["lin_fit"] = linear.best_fit
                data["lin_conf_lower"] = data["lin_fit"] - lin_conf
                data["lin_conf_upper"] = data["lin_fit"] + lin_conf

            if s.quad_fit_show:
                quadratic = quadratic_fit(data["percent_size_reduction"], data["metric"])
                quad_conf = quadratic.eval_uncertainty(x=data["percent_size_reduction"], sigma=2)
                data["quad_fit"] = quadratic.best_fit
                data["quad_conf_lower"] = data["quad_fit"] - quad_conf
                data["quad_conf_upper"] = data["quad_fit"] + quad_conf

            source = ColumnDataSource(data)

            if s.lin_fit_show:
                fig.line(x="percent_size_reduction", y="lin_fit",
                         legend_label=f"{"outdoor"} Linear fit", source=source, color=cc.b_glasbey_hv[index])
                lin_band = Band(base="percent_size_reduction", lower="lin_conf_lower", upper="lin_conf_upper",
                                source=source, fill_color=cc.b_glasbey_hv[index], fill_alpha=0.5)
                fig.add_layout(lin_band)

            if s.quad_fit_show:
                fig.line(x="percent_size_reduction", y="quad_fit",
                         legend_label=f"{"outdoor"} Quadratic fit", source=source, color=cc.b_glasbey_hv[index])
                quad_band = Band(base="percent_size_reduction", lower="quad_conf_lower", upper="quad_conf_upper",
                                 source=source, fill_color=cc.b_glasbey_hv[index], fill_alpha=0.5)
                fig.add_layout(quad_band)

        return fig

    def __panel__(self):
        series_input = SeriesInput()
        pn.bind(self.value.add_series, series_input.param.value, watch=True)
        return pn.Row(series_input, self._layout)


series_list = SeriesPlots(value=[])
plot_editor = PlotEditor(value=series_list)

template = pn.template.BootstrapTemplate(title="RAISE Explorer")
template.main.append(plot_editor)

home_button = pn.widgets.Button(name="Index", button_type="primary")
home_button.js_on_click(code="window.location.href='/'")
template.header.append(home_button)

# bokeh_pane = pn.bind(plot, series_list.param.ls)

# template.main.append(bokeh_pane)

# table = pn.widgets.Tabulator(data)
# template.main.append(table)

template.servable()
