import actxps as xp
from shiny import ui, render, reactive, App

import pandas as pd
import numpy as np
from warnings import warn
from datetime import datetime
from copy import deepcopy
from plotnine import (aes,
                      geom_smooth,
                      theme_light,
                      theme,
                      element_text,
                      element_rect)
import io


def _exp_shiny(obj,
               predictors=None,
               expected=None,
               distinct_max=25):
    """
    Internal function for creating interactive shiny apps. This function is 
    not meant to be called directly. Use `ExposedDF.exp_shiny()` instead.
    """

    from actxps import ExposedDF
    assert isinstance(obj, ExposedDF)

    dat = obj.data
    cols = dat.columns

    # convert boolean columns to strings
    obj.data[obj.data.select_dtypes(bool).columns] = \
        obj.data.select_dtypes(bool).apply(lambda x: x.astype(str))

    if predictors is None:
        predictors = cols
    else:
        predictors = pd.Index(np.atleast_1d(predictors))

    if expected is None:
        expected = cols[cols.str.contains('expected')]
    else:
        expected = pd.Index(np.atleast_1d(expected))

    # check for presence of transactions
    has_trx = len(obj.trx_types) > 0
    trx_cols = cols[cols.str.contains("^trx_(?:n|amt)_", regex=True)]

    if any(~(predictors.append(expected)).isin(cols)):
        warn("All predictors and expected values must be columns in `dat`. " +
             "Unexpected values will be removed.")
        predictors = predictors[predictors.isin(cols)]
        expected = expected[expected.isin(cols)]

    predictors = predictors.tolist()
    expected = expected.tolist()

    total_rows = len(dat)

    # organize predictors
    preds = pd.DataFrame({'predictors': predictors})
    # drop non-predictors (if any)
    non_preds = ["pol_num", "status", "term_date", "exposure"]
    if has_trx:
        non_preds.extend(trx_cols)
    preds = preds.loc[~preds.predictors.isin(non_preds)]

    preds['dtype'] = dat[preds.predictors].dtypes.to_numpy()
    preds['dtype'] = preds['dtype'].apply(str)
    preds['is_number'] = preds.dtype.str.contains('(?:int|float)')

    dtype_cond = [
        preds.dtype.str.contains('date'),
        (preds.dtype == 'category') | (preds.dtype == 'object'),
        preds.is_number,
        np.repeat(True, len(preds))
    ]
    dtype_labels = ['date', 'category', 'number', 'other']
    preds['order'] = np.select(dtype_cond, range(len(dtype_cond)))
    preds.dtype = np.select(dtype_cond, dtype_labels)
    preds['n_unique'] = preds.predictors.apply(
        lambda x: len(dat[x].unique())
    )
    preds = (preds.
             sort_values('order').
             drop(columns='order').
             set_index('predictors'))
    preds_small = preds.copy()[preds.n_unique <= distinct_max].index.to_list()

    yVar_exp = ["q_obs", "n_claims", "claims", "exposure", "credibility"]

    if has_trx:
        yVar_trx = ["trx_util", "trx_freq", "trx_n", "trx_flag",
                    "trx_amt", "avg_trx", "avg_all"]
        available_studies = {"Termination study": "exp",
                             "Transaction study": "trx"}
    else:
        yVar_trx = None
        available_studies = {"Termination study": "exp"}

    # function to make input widgets
    def widget(x, checkbox_limit=8):

        inputId = "i_" + x

        assert x in dat.columns, \
            f"Error creating an input widget for {x}. " + \
            f"{x} does not exist in the input data."

        if preds.loc[x]['dtype'] == "number":

            min_val = dat[x].min()
            max_val = dat[x].max()

            inp = ui.input_slider(
                inputId, ui.strong(x),
                min=min_val,
                max=max_val,
                value=(min_val, max_val)
            )

        elif preds.loc[x]['dtype'] == "date":

            def fmt_date(x):
                return datetime.strftime(x, "%Y-%m-%d")

            min_val = fmt_date(dat[x].min())
            max_val = fmt_date(dat[x].max())

            inp = ui.input_date_range(
                inputId, ui.strong(x),
                start=min_val,
                end=max_val,
                min=min_val,
                max=max_val,
                startview="year"
            )

        elif preds.loc[x]['dtype'] == 'category':

            choices = dat[x].unique().tolist()

            if len(choices) > checkbox_limit:
                inp = ui.input_select(
                    inputId, ui.strong(x),
                    choices=choices, selected=choices,
                    multiple=True
                )
            else:
                inp = ui.input_checkbox_group(
                    inputId, ui.strong(x),
                    choices=choices, selected=choices
                )

        else:
            raise TypeError(f"Error creating an input widget for {x}. " +
                            f"{x} is of class {str(dat[x].dtype)}, " +
                            "which is not supported.")

        return inp

    def widgetPred(fun):
        def new_fun(inputId, label, width, choices=None, **kwargs):

            if choices is None:
                choices = ["None"] + preds.index.to_list()

            return ui.column(
                width,
                fun(inputId, ui.strong(label), choices=choices, **kwargs)
            )
        return new_fun

    selectPred = widgetPred(ui.input_select)
    checkboxGroupPred = widgetPred(ui.input_checkbox_group)

    # expected values set up
    if len(expected) > 0:
        has_expected = True
        expected_widget = checkboxGroupPred(
            "ex_checks", "Expected values:", 4,
            choices=expected,
            selected=expected)
    else:
        has_expected = False
        expected_widget = None

    # transactions set up
    if has_trx:

        percent_of_choices = preds.loc[preds.is_number].index.to_list()

        trx_tab = ui.nav(
            "Transaction study",
            ui.row(
                checkboxGroupPred(
                    "trx_types_checks",
                    "Transaction types:", 4,
                    obj.trx_types,
                    selected=obj.trx_types),
                selectPred(
                    "pct_checks",
                    "Transactions as % of:", 4,
                    choices=percent_of_choices,
                    multiple=True),
                ui.column(
                    4,
                    ui.input_checkbox("trx_combine",
                                      ui.strong("Combine transactions?"),
                                      False)
                )
            ),
            value="trx"
        )

    else:
        trx_tab = None

    app_ui = ui.page_fluid(

        ui.panel_title(("/".join(obj.target_status) + " Experience Study" +
                        ((" and " + "/".join(obj.trx_types) +
                          " Transaction Study")
                         if has_trx else ""))
                       ),

        ui.layout_sidebar(

            ui.panel_sidebar(
                ui.h3("Filters"),
                [widget(x) for x in preds.index]),

            ui.panel_main(
                ui.panel_well(
                    ui.h3("Study options"),

                    ui.h4("Grouping variables"),
                    ui.em(("The variables selected below will be used as " +
                           "grouping variables in the plot and table outputs." +
                           " Multiple variables can be selected as facets.")),
                    ui.row(
                        selectPred("xVar", "x:", 4),
                        selectPred("colorVar", "Color:", 4,
                                   choices=["None"] + preds_small),
                        selectPred("facetVar", "Facets:", 4, multiple=True,
                                   choices=preds_small)
                    ),

                    ui.h4("Study type"),
                    ui.navset_pill(
                        ui.nav("Termination study",
                               ui.row(
                                   expected_widget,
                                   selectPred("weightVar",
                                              "Weight by:", 4,
                                              choices=["None"] +
                                              preds.loc[preds.is_number].index.to_list(
                                              )
                                              )
                               ),
                               value="exp"
                               ),
                        trx_tab,
                        id="study_type"
                    )

                ),

                ui.h3("Output"),

                ui.navset_pill(
                    ui.nav(
                        "Plot",
                        ui.br(),
                        ui.row(
                            selectPred("yVar", "y:", 4, choices=yVar_exp),
                            ui.column(
                                4,
                                ui.input_radio_buttons(
                                    "plotGeom",
                                    ui.strong("Geometry:"),
                                    {"bars": "Bars",
                                     "lines": "Lines and Points"})
                            ),
                            ui.column(
                                4,
                                ui.input_checkbox(
                                    "plotSmooth",
                                    ui.strong("Add Smoothing?"),
                                    False)
                            )
                        ),

                        ui.row(
                            ui.column(
                                4,
                                ui.input_checkbox(
                                    "plotFreeY",
                                    ui.strong("Free y Scales?"),
                                    False)
                            )
                        ),

                        ui.output_plot("xpPlot")
                    ),

                    ui.nav(
                        "Table",
                        ui.br(),
                        ui.output_table("xpTable")
                    ),

                    ui.nav(
                        "Export Data",
                        ui.br(),
                        ui.download_button("xpDownload", "Download")
                    )

                ),

                ui.h3("Filter information"),
                ui.output_text_verbatim("filterInfo")

            )
        )

    )

    def server(input, output, session):

        @reactive.Calc
        def yVar_exp2():
            return (yVar_exp +
                    list(input.ex_checks()) +
                    [f"ae_{x}" for x in input.ex_checks()] +
                    [f"adj_{x}" for x in input.ex_checks()])

        @reactive.Calc
        def yVar_trx2():
            return (yVar_trx +
                    [f"pct_of_{x}_w_trx" for x in input.pct_checks()] +
                    [f"pct_of_{x}_all" for x in input.pct_checks()] +
                    list(input.pct_checks()) +
                    [f"{x}_w_trx" for x in input.pct_checks()])

        # update y variable selections in response to inputs
        @reactive.Effect
        @reactive.event(input.study_type, input.ex_checks, input.pct_checks)
        def _():
            if input.study_type() == "exp":
                choices = yVar_exp2()
            else:
                choices = yVar_trx2()

            ui.update_select(
                "yVar",
                choices=choices
            )

        # reactive data
        @reactive.Calc
        def rdat():

            # function to build filter expressions
            def expr_filter(x):

                inp_val = input["i_" + x]()

                # ensure that dates are quoted
                if preds.loc[x]['dtype'] == 'date':
                    inp_val = [f"'{a}'" for a in inp_val]

                # create filter expressions
                if preds.loc[x]['dtype'] in ['date', 'number']:
                    res = f"({x} >= {inp_val[0]}) & ({x} <= {inp_val[1]})"
                else:
                    # categorical
                    res = f"({x}.isin({inp_val}))"

                return res

            filters = [expr_filter(x) for x in preds.index]
            filters = ' & '.join(filters)

            new_obj = deepcopy(obj)
            new_obj.data = new_obj.data.query(filters)

            return new_obj

        # experience study
        @reactive.Calc
        def rxp():

            obj = rdat()

            groups = [input.xVar(), input.colorVar()] + \
                list(input.facetVar())
            groups = [x for x in groups if x != "None"]
            # ensure uniqueness
            groups = list(set(groups))

            if input.weightVar() == "None":
                wt = None
            else:
                wt = input.weightVar()
                assert wt not in groups, \
                    "Error: the weighting variable cannot be one of the " + \
                    "grouping (x, color, facets) variables."

            if has_expected:
                ex = list(input.ex_checks())
            else:
                ex = None

            if input.study_type() == "exp":
                return (obj.
                        groupby(*groups).
                        exp_stats(wt=wt,
                                  credibility=True,
                                  expected=ex))
            else:
                return (obj.
                        groupby(*groups).
                        trx_stats(percent_of=list(input.pct_checks()),
                                  trx_types=list(input.trx_types_checks()),
                                  combine_trx=input.trx_combine()))

        @output()
        @render.plot()
        def xpPlot():

            if (input.study_type() == "exp") & (input.yVar() in yVar_trx2()):
                return None
            if (input.study_type() == "trx") & (input.yVar() in yVar_exp2()):
                return None

            new_rxp = deepcopy(rxp())

            if input.xVar() != "None":
                x = input.xVar()
            else:
                new_rxp.data["All"] = ""
                x = "All"

            y = input.yVar()

            if input.colorVar() != "None":
                color = input.colorVar()
                mapping = aes(x, y, color=color,
                              fill=color, group=color)
            else:
                color = None
                mapping = aes(x, y)

            # y labels
            if input.yVar() in (["claims", "n_claims", "exposure",
                                "trx_n", "trx_flag", "trx_amt",
                                 "avg_trx", "avg_all"] +
                                list(input.pct_checks()) +
                                [i + "_w_trx" for i in input.pct_checks()]):
                def y_labels(l): return [f"{v:,.0f}" for v in l]
            elif input.yVar() == "trx_freq":
                def y_labels(l): return [f"{v:,.1f}" for v in l]
            else:
                def y_labels(l): return [f"{v * 100:.1f}%" for v in l]

            if len(input.facetVar()) == 0:
                p = new_rxp.plot(mapping=mapping, geoms=input.plotGeom(),
                                 y_labels=y_labels)
            else:

                facets = list(input.facetVar())
                if input.study_type() == "trx":
                    facets.append("trx_type")

                p = new_rxp.plot(mapping=mapping,
                                 geoms=input.plotGeom(),
                                 y_labels=y_labels,
                                 facets=facets,
                                 scales="free_y" if input.plotFreeY() else "fixed")

            if input.plotSmooth():
                p = p + geom_smooth(method="loess")

            return (p +
                    theme_light() +
                    theme(axis_text=element_text(size=10),
                          strip_text=element_text(size=12),
                          strip_background=element_rect(fill="#43536b"),
                          dpi=300
                          )
                    )

        @output
        @render.table
        def xpTable():
            return (rxp().
                    table().
                    set_table_attributes('class="dataframe table shiny-table w-auto"'))

        # filter information
        @output
        @render.text
        def filterInfo():

            curr_rows = len(rdat().data)

            return (f"Total records = {total_rows:,d}\n" +
                    f"Remaining records = {curr_rows:,d}\n" +
                    f"% Data Remaining = {curr_rows / total_rows * 100:,.1f}%")

        # download data
        @session.download(
            # TODO should this use tepmfile.TemporaryDirectory?
            filename=lambda: f"{input.study_type()}-data-{datetime.today().isoformat(timespec='minutes')[:10]}.csv"
        )
        def xpDownload():
            with io.BytesIO() as buf:
                rxp().data.to_csv(buf)
                yield buf.getvalue()

    # Run the application
    return App(app_ui, server)
