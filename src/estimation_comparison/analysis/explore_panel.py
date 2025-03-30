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
from bokeh.models import ColumnDataSource, Band, Range1d
from bokeh.plotting import figure

from estimation_comparison.analysis.fit import quadratic_fit, linear_fit
from estimation_comparison.analysis.panel_model import Series
from estimation_comparison.database import BenchmarkDatabase

db = BenchmarkDatabase(Path("benchmarks/benchmark.sqlite"))


class SeriesPlots(pn.viewable.Viewer):
    value = param.List(item_type=Series)

    def __init__(self, **params):
        super().__init__(**params)

    def add_series(self, series):
        self.value = [*self.value, series]

    def clear_series(self, *_):
        self.value = []

    def pop_series(self, *_):
        self.value = self.value[:-1]


class SeriesInput(pn.viewable.Viewer):
    value: Series = param.ClassSelector(class_=Series)

    def __panel__(self):
        pre_select = pn.widgets.Select(name="Preprocessor", options=[entry[1] for entry in db.get_preprocessors()])
        est_select = pn.widgets.Select(name="Estimator", options=[entry[1] for entry in db.get_estimators()])
        comp_select = pn.widgets.Select(name="Compressor", options=[entry[1] for entry in db.get_compressors()])
        block_select = pn.widgets.Select(name="Block Summary",
                                         options=[entry[1] for entry in db.get_block_summary_funcs()] + ["None"])
        file_select = pn.widgets.Select(name="File Summary",
                                        options=[entry[1] for entry in db.get_file_summary_funcs()] + ["None"])

        lin_fit_show = pn.widgets.Checkbox(name="Show linear fit", value=False)
        quad_fit_show = pn.widgets.Checkbox(name="Show quadratic fit", value=False)

        add_button = pn.widgets.Button(name="Add", button_type="primary", align="center")
        plot_controls = pn.layout.WidgetBox("## Configure series", pre_select, est_select, comp_select, block_select,
                                            file_select, lin_fit_show, quad_fit_show, add_button)

        @pn.depends(pre_select, est_select, comp_select, block_select, file_select, lin_fit_show, quad_fit_show,
                    add_button, watch=True)
        def create_series(pre_select, est_select, comp_select, block_select, file_select, lin_fit_show, quad_fit_show,
                          add_button):
            if add_button:
                self.value = Series(preprocessor=pre_select, estimator=est_select,
                                    compressor=comp_select, block_summary_fn=block_select, file_summary_fn=file_select,
                                    lin_fit_show=lin_fit_show, quad_fit_show=quad_fit_show)

        return plot_controls


class PlotEditor(pn.viewable.Viewer):
    value: SeriesPlots = param.ClassSelector(class_=SeriesPlots)

    @param.depends("value.value")
    def _layout(self):
        series = self.value.value
        return self._plot(series)

    @staticmethod
    def _plot(series):
        fig = figure(x_range=Range1d(0, 100))
        fig.sizing_mode = "scale_both"

        for index, s in enumerate(series):
            desc, rec = db.get_solo_plot_dataframe(s.preprocessor, s.estimator, s.compressor, s.block_summary_fn,
                                                   s.file_summary_fn)

            data = pd.DataFrame.from_records(rec, columns=[item[0] for item in desc])

            data["percent_size_reduction"] = (1.0 - (data["final_size"] / data["initial_size"])) * 100.0
            data.sort_values("percent_size_reduction", inplace=True)
            fig.scatter(x="percent_size_reduction", y="metric",
                        legend_label=f"{s.estimator} ({s.preprocessor}), {s.compressor}",
                        color=cc.b_glasbey_hv[index], source=data, alpha=0.2)

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
                fig.line(x="percent_size_reduction", y="lin_fit", source=source, color=cc.b_glasbey_hv[index])
                lin_band = Band(base="percent_size_reduction", lower="lin_conf_lower", upper="lin_conf_upper",
                                source=source, fill_color=cc.b_glasbey_hv[index], fill_alpha=0.5)
                fig.add_layout(lin_band)

            if s.quad_fit_show:
                fig.line(x="percent_size_reduction", y="quad_fit", source=source, color=cc.b_glasbey_hv[index])
                quad_band = Band(base="percent_size_reduction", lower="quad_conf_lower", upper="quad_conf_upper",
                                 source=source, fill_color=cc.b_glasbey_hv[index], fill_alpha=0.5)
                fig.add_layout(quad_band)

        return fig

    def __panel__(self):
        series_input = SeriesInput()

        remove_all_button = pn.widgets.Button(name="Remove All", button_type="primary")
        remove_last_button = pn.widgets.Button(name="Remove Last", button_type="primary")

        pn.bind(self.value.add_series, series_input.param.value, watch=True)
        pn.bind(self.value.pop_series, remove_last_button, watch=True)
        pn.bind(self.value.clear_series, remove_all_button, watch=True)

        button_row = pn.Row(remove_last_button, remove_all_button, align="center")

        controls = pn.Column(series_input, button_row)

        return pn.Row(controls, self._layout)


series_list = SeriesPlots(value=[])
plot_editor = PlotEditor(value=series_list)

template = pn.template.BootstrapTemplate(title="RAISE Explorer")
template.main.append(plot_editor)

home_button = pn.widgets.Button(name="Index", button_type="primary")
home_button.js_on_click(code="window.location.href='/panel/'")
template.header.append(home_button)

template.servable()
