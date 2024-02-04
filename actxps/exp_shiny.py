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
from actxps.col_select import col_contains
import io


def exp_shiny(self,
              predictors=None,
              expected=None,
              distinct_max=25,
              col_exposure='exposure'):
    """
    Interactively explore experience data

    Launch a shiny application to interactively explore drivers of
    experience.

    Parameters
    ----------
    predictors : str | list | np.ndarray, default=`None`
        A character vector of independent variables in the `data` property 
        to include in the shiny app.
    expected : str | list | np.ndarray, default=`None`
        A character vector of expected values in the `data` property to
        include in the shiny app.
    distinct_max : int
        Maximum number of distinct values allowed for `predictors`
        to be included as "Color" and "Facets" grouping variables. This 
        input prevents the drawing of overly complex plots. Default 
        value = 25.

    Notes
    ----------
    If transactions have been attached to the `ExposedDF` object, the app
    will contain features for both termination and transaction studies.
    Otherwise, the app will only support termination studies.

    If nothing is passed to `predictors`, all columns names in `dat` will be
    used (excluding the policy number, status, termination date, exposure,
    transaction counts, and transaction amounts columns).

    The `expected` argument is optional. As a default, any column names
    containing the word "expected" are used.

    **Layout**

    *Filters*

    The sidebar contains filtering widgets for all variables passed
    to the `predictors` argument.

    *Study options*

    Grouping variables

    This box includes widgets to select grouping variables for summarizing
    experience. The "x" widget is also used as the x variable in the plot
    output. Similarly, the "Color" and "Facets" widgets are used for color
    and facets in the plot. Multiple faceting variables are allowed. For 
    the table output, "x", "Color", and "Facets" have no particular meaning 
    beyond the order in which of grouping variables are displayed.

    Study type

    This box also includes a toggle to switch between termination studies 
    and transaction studies (if available).

    - Termination studies:
    The expected values checkboxes are used to activate and deactivate
    expected values passed to the `expected` argument. This impacts the
    table output directly and the available "y" variables for the plot. If
    there are no expected values available, this widget will not appear.
    The "Weight by" widget is used to specify which column, if any, 
    contains weights for summarizing experience.

    - Transaction studies:
    The transaction types checkboxes are used to activate and deactivate
    transaction types that appear in the plot and table outputs. The
    available transaction types are taken from the `trx_types` property of 
    the `ExposedDF` object. In the plot output, transaction type will 
    always appear as a faceting variable. The "Transactions as % of"
    selector will expand the list of available "y" variables for the plot 
    and impact the table output directly. Lastly, a checkbox exists that 
    allows for all transaction types to be aggregated into a single group.

    **Output**

    *Plot Tab*

    This tab includes a plot and various options for customization:

    - y: y variable
    - Geometry: plotting geometry
    - Add Smoothing?: activate to plot loess curves
    - Free y Scales: activate to enable separate y scales in each plot.

    *Table*

    This tab includes a data table.

    *Export Data*

    This tab includes a download button that will save a copy of the 
    summarized experience data.

    **Filter Information**

    This box contains information on the original number of exposure 
    records, the number of records after filters are applied, and the 
    percentage of records retained.

    Examples 
    ----------
    ```{python}
    import actxps as xp
    import numpy as np

    census_dat = xp.load_census_dat()
    withdrawals = xp.load_withdrawals()
    account_vals = xp.load_account_vals()

    expo = xp.ExposedDF(census_dat, "2019-12-31",
                        target_status = "Surrender")
    expected_table = np.concatenate((np.linspace(0.005, 0.03, 10),
                                    [.2, .15], np.repeat(0.05, 3)))
    expo.data['expected_1'] = expected_table[expo.data.pol_yr - 1]
    expo.data['expected_2'] = np.where(expo.data.inc_guar, 0.015, 0.03)
    expo.add_transactions(withdrawals)
    expo.data = expo.data.merge(account_vals, how='left',
                                on=["pol_num", "pol_date_yr"])

    app = expo.exp_shiny(expected=['expected_1', 'expected_2'])
    ```
    """
    # check that shiny is installed
    try:
        from shiny import ui, render, reactive, App
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The 'shiny' package is required to " + 
                                  "use this function")
    try:
        from shinyswatch import theme
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The 'shinyswatch' package is required to " +
                                  "use this function")        

    from actxps import SplitExposedDF
    from actxps.expose_split import _check_split_expose_basis

    # special logic required for split exposed data frames
    if isinstance(self, SplitExposedDF):
        _check_split_expose_basis(self, col_exposure)
        dat = dat.rename(columns={col_exposure: 'exposure'})

        if col_exposure == "exposure_cal":
            dat.drop(columns=['exposure_pol'], inplace=True)
        else:
            dat.drop(columns=['exposure_cal'], inplace=True)

    # make a copy of the ExposedDF to avoid unexpected changes
    expo = deepcopy(self)
    dat = expo.data
    cols = dat.columns

    # convert boolean columns to strings
    dat[dat.select_dtypes(bool).columns] = \
        dat.select_dtypes(bool).apply(lambda x: x.astype(str))

    if predictors is None:
        predictors = cols
    else:
        predictors = pd.Index(np.atleast_1d(predictors))

    if expected is None:
        expected = col_contains(dat, 'expected')
    else:
        expected = pd.Index(np.atleast_1d(expected))

    # check for presence of transactions
    has_trx = len(expo.trx_types) > 0
    trx_cols = col_contains(dat, "^trx_(?:n|amt)_")

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
    preds['is_integer'] = preds.dtype.str.contains('(?:int)')

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
    
    def calc_scope(p, c):
        if (c in ['date', 'number']):
            return min(dat[p]), max(dat[p])
        else:
            return dat[p].unique()
    preds['scope'] = preds[['predictors', 'dtype']].\
        apply(lambda x: calc_scope(x['predictors'], x['dtype']), axis = 1)
    
    preds = (preds.
             sort_values('order').
             drop(columns='order').
             set_index('predictors'))
    preds_small = preds.copy()[preds.n_unique <= distinct_max].index.to_list()

    yVar_exp = ["q_obs", "n_claims", "claims", "exposure", "credibility"]

    if has_trx:
        yVar_trx = ["trx_util", "trx_freq", "trx_n", "trx_flag",
                    "trx_amt", "avg_trx", "avg_all"]
    else:
        yVar_trx = None

    # function to make input widgets
    def widget(x, checkbox_limit=8):

        inputId = "i_" + x

        assert x in dat.columns, \
            f"Error creating an input widget for {x}. " + \
            f"{x} does not exist in the input data."
            
        info = preds.loc[x]
        choices = info['scope']

        if info['dtype'] == "number":

            inp = ui.input_slider(
                inputId, ui.strong(x),
                min=choices[0],
                max=choices[1],
                value=choices,
                step = 1 if info['is_integer'] and info['n_unique'] < 100 \
                    else None
            )

        elif info['dtype'] == "date":

            def fmt_date(x):
                return datetime.strftime(x, "%Y-%m-%d")

            min_val = fmt_date(choices[0])
            max_val = fmt_date(choices[1])

            inp = ui.input_date_range(
                inputId, ui.strong(x),
                start=min_val,
                end=max_val,
                min=min_val,
                max=max_val,
                startview="year"
            )

        elif info['dtype'] == 'category':

            choices = choices.tolist()

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
                    expo.trx_types,
                    selected=expo.trx_types),
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

        ui.panel_title(("/".join(expo.target_status) + " Experience Study" +
                        ((" and " + "/".join(expo.trx_types) +
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
                                     "lines": "Lines and Points",
                                     "points": "Points"})
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

            new_expo = deepcopy(expo)
            new_expo.data = new_expo.data.query(filters)

            return new_expo

        # experience study
        @reactive.Calc
        def rxp():

            expo = rdat()

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
                return (expo.
                        groupby(*groups).
                        exp_stats(wt=wt,
                                  credibility=True,
                                  expected=ex))
            else:
                return (expo.
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
