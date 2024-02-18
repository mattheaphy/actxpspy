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
import matplotlib.pyplot as plt
import io
import great_tables.shiny as gts


def exp_shiny(self,
              predictors: str | list | np.ndarray = None,
              expected: str | list | np.ndarray = None,
              distinct_max: int = 25,
              title: str = None,
              credibility: bool = True,
              conf_level: float = 0.95,
              cred_r: float = 0.05,
              bootswatch_theme: str = None,
              col_exposure: str = 'exposure'):
    """
    Interactively explore experience data

    Launch a Shiny application to interactively explore drivers of
    experience.

    Parameters
    ----------
    predictors : str | list | np.ndarray, default=None
        A character vector of independent variables in the `data` property 
        to include in the shiny app.
    expected : str | list | np.ndarray, default=None
        A character vector of expected values in the `data` property to
        include in the shiny app.
    distinct_max : int
        Maximum number of distinct values allowed for `predictors`
        to be included as "Color" and "Facets" grouping variables. This 
        input prevents the drawing of overly complex plots. Default 
        value = 25.
    title : str, default=None
        Title of the Shiny app. If no title is provided, a descriptive title 
        will be generated based on attributes of the `ExposedDF` object.
    credibility : bool, default=False
        If `True`, future calls to `summary()` will include partial 
        credibility weights and credibility-weighted termination rates.
    conf_level : float, default=0.95
        Confidence level used for the Limited Fluctuation credibility method
        and confidence intervals.
    cred_r : float, default=0.05
        Error tolerance under the Limited Fluctuation credibility method.
    bootswatch_theme : str, default=None
        The name of a preset bootswatch theme passed to shinyswatch.get_theme.
    col_exposure : str, default='exposure'
        Name of the column in the `data` property containing exposures. This
        input is only used to clarify the exposure basis when the `ExposedDF` 
        is also a `SplitExposedDF` object. For more information on split 
        exposures, see ExposedDF.expose_split().


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

    The sidebar contains filtering widgets organized by data type for all
    variables passed to the `predictors` argument.

    At the top of the sidebar, information is shown on the percentage of records
    remaining after applying filters. A description of all active filters is 
    also provided.

    The top of the sidebar also includes a "play / pause" switch that can pause
    reactivity of the application. Pausing is a good option when multiple 
    changes are made in quick succession, especially when the underlying data 
    set is large.

    *Grouping variables*

    This box includes widgets to select grouping variables for summarizing
    experience. The "x" widget is also used as the x variable in the plot
    output. Similarly, the "Color" and "Facets" widgets are used for color
    and facets. Multiple faceting variables are allowed. For the table output,
    "x", "Color", and "Facets" have no particular meaning beyond the order in 
    which of grouping variables are displayed.

    *Study type*

    This box includes a toggle to switch between termination studies and
    transaction studies (if available). Different options are available for each
    study type.

    Termination studies

    The expected values checkboxes are used to activate and deactivate
    expected values passed to the `expected` argument. This impacts the
    table output directly and the available "y" variables for the plot. If
    there are no expected values available, this widget will not appear.
    The "Weight by" widget is used to specify which column, if any, 
    contains weights for summarizing experience.

    Transaction studies

    The transaction types checkboxes are used to activate and deactivate
    transaction types that appear in the plot and table outputs. The
    available transaction types are taken from the `trx_types` property of 
    the `ExposedDF` object. In the plot output, transaction type will 
    always appear as a faceting variable. The "Transactions as % of"
    selector will expand the list of available "y" variables for the plot 
    and impact the table output directly. Lastly, a checkbox exists that 
    allows for all transaction types to be aggregated into a single group.

    *Output*

    Plot

    This tab includes a plot and various options for customization:

    - y: y variable
    - Geometry: plotting geometry
    - Add Smoothing: activate to plot loess curves
    - Confidence intervals: If available, add error bars for confidence 
      intervals around the selected y variable
    - Free y Scales: activate to enable separate y scales in each plot
    - Log y-axis: activate to plot all y-axes on a log-10 scale

    The gear icon above the plot contains a pop-up menu that can be used to
    change the size of the plot for exporting.

    Table

    The gear icon above the table contains a pop-up menu that can be used to
    change the appearance of the table:

    - The "Confidence intervals" and "Credibility-weighted termination rates"
    switches add these outputs to the table. These values are hidden as a 
    default to prevent over-crowding.
    - The "Include color scales" switch disables or re-enables conditional color
    formatting.
    - The "Decimals" slider controls the number of decimals displayed for
    percentage fields.
    - The "Font size multiple" slider impacts the table's font size

    Export

    This pop-up menu contains options for saving summarized experience data or 
    the plot. Data is saved as a CSV file. The plot is saved as a png file.


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
    # check that required packages are installed
    try:
        from shiny import ui, render, reactive, App
        from faicons import icon_svg
        from shinyswatch import get_theme
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The 'shiny, faicons, and shinyswatch " +
                                  "packages are required to " +
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
        expected = pd.Index(col_contains(dat, 'expected'))
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
    preds['is_number'] = preds['dtype'].str.contains('(?:int|float)')
    preds['is_integer'] = preds['dtype'].str.contains('(?:int)')

    dtype_cond = [
        preds['dtype'].str.contains('date'),
        (preds['dtype'] == 'category') | (preds['dtype'] == 'object'),
        preds.is_number,
        np.repeat(True, len(preds))
    ]
    dtype_labels = ['Dates', 'Categorical', 'Numeric', 'other']
    preds['order'] = np.select(dtype_cond, range(len(dtype_cond)))
    preds['dtype'] = np.select(dtype_cond, dtype_labels)
    preds['n_unique'] = preds.predictors.apply(
        lambda x: len(dat[x].unique())
    )

    def calc_scope(p, c):
        if (c in ['Dates', 'Numeric']):
            return min(dat[p]), max(dat[p])
        else:
            return dat[p].unique()
    preds['scope'] = preds[['predictors', 'dtype']].\
        apply(lambda x: calc_scope(x['predictors'], x['dtype']), axis=1)

    preds = (preds.
             sort_values('order').
             drop(columns='order').
             set_index('predictors'))
    preds_small = preds.copy()[preds.n_unique <= distinct_max].index.to_list()

    yVar_exp = ["q_obs", "n_claims", "claims", "exposure"]
    if credibility:
        yVar_exp.append("credibility")

    if has_trx:
        yVar_trx = ["trx_util", "trx_freq", "trx_n", "trx_flag",
                    "trx_amt", "avg_trx", "avg_all", "exposure"]
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

        if info['dtype'] == "Numeric":

            inp = ui.input_slider(
                inputId, ui.strong(x),
                min=choices[0],
                max=choices[1],
                value=choices,
                step=1 if info['is_integer'] and info['n_unique'] < 100
                else None
            )

        elif info['dtype'] == "Dates":

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

        elif info['dtype'] == 'Categorical':

            choices = choices.tolist()

            if len(choices) > checkbox_limit:
                inp = ui.input_selectize(
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

    selectPred = widgetPred(ui.input_selectize)
    checkboxGroupPred = widgetPred(ui.input_checkbox_group)

    # create a tooltip with an info icon
    def info_tooltip(*args, **kwargs):
        return ui.tooltip(icon_svg("circle-info"), *args, **kwargs)

    # expected values set up
    if len(expected) > 0:
        has_expected = True
        expected_widget = checkboxGroupPred(
            "ex_checks", "Expected values:", 6,
            choices=expected,
            selected=expected)
    else:
        has_expected = False
        expected_widget = None

    # set up nav panels for study type options
    study_type_nav_panels = [
        ui.nav_panel(
            ["Termination study ",
             info_tooltip(
                 """
                         Choose expected values (if available) that appear in 
                         the plot and table outputs. If desired, select a 
                         weighting variable for summarizing experience.
                         """)
             ],

            ui.row(
                expected_widget,
                selectPred("weightVar",
                           "Weight by:", 4,
                           choices=["None"] +
                           preds.loc[preds.is_number].index.to_list(
                           ))
            ),

            value="exp")
    ]

    # transactions set up
    if has_trx:

        percent_of_choices = preds.loc[preds.is_number].index.to_list()

        trx_tab = ui.nav_panel(
            ["Transaction study ",
             info_tooltip(
                 'Choose transaction types and "percent of" variables that appear in \
                     the plot and table outputs. If desired, combine all transaction types \
                         into a single group.')],
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
                    ui.input_switch("trx_combine",
                                    ui.strong("Combine transactions?"),
                                    False)
                )
            ),
            value="trx"
        )
        study_type_nav_panels.append(trx_tab)

    if title is None:
        title = ("/".join(expo.target_status) + " Experience Study" +
                 ((" and " + "/".join(expo.trx_types) +
                   " Transaction Study")
                  if has_trx else ""))

    app_ui = ui.page_sidebar(

        ui.sidebar(
            ui.input_switch("play",
                            ["Reactivity ",
                             icon_svg("play"), "/",
                             icon_svg("pause")],
                            value=True),
            # filter descriptions
            ui.tooltip(
                ui.value_box(
                    title="% data remaining",
                    theme="primary",
                    value=ui.output_text("rem_pct"),
                    showcase=ui.output_plot("filter_pie",
                                            width="75px", height="75px")
                ),
                f"Original row count: {total_rows:,d}",
                ui.output_text("rem_rows")
            ),

            ui.strong(ui.output_text("filter_desc_header")),
            ui.tags.small(ui.output_text_verbatim('filter_desc')),

            # add filter widgets
            ui.accordion(
                *map(lambda x: ui.accordion_panel(
                    x[0],
                    [widget(y) for y in x[1].index]),
                    preds.groupby('dtype', observed=True, sort=False)),
                open=True
            ),

            width=370,
            title="Filters"
        ),

        ui.layout_column_wrap(

            ui.card(
                ui.card_header(
                    "Grouping variables ",
                    info_tooltip(
                        """
                        The variables selected below will be used as
                        grouping variables in the plot and table outputs.
                        Multiple variables can be selected as facets.
                        """
                    )),

                ui.row(
                    selectPred("xVar", "x:", 4),
                    selectPred("colorVar", "Color:", 4,
                               choices=["None"] + preds_small),
                    selectPred("facetVar", "Facets:", 4, multiple=True,
                               choices=preds_small)
                ),
            ),

            ui.navset_card_tab(
                *study_type_nav_panels,
                id="study_type",
                title="Study type",
            ),
            width=400,
            heights_equal='row'
        ),

        ui.navset_bar(
            ui.nav_panel(
                "Plot",
                ui.card(
                    ui.card_header(
                        "Plot inputs ",
                        info_tooltip(
                            ui.markdown(
                                """
                            <div style="text-align: left">
                            
                            - `y`-axis variable selection
                            - `Second y-axis` toggle and variable
                            - `Geometry` for plotting
                            - `Add smoothing`: add smooth loess curves
                            - `Confidence intervals`: If available, draw confidence interval
                            error bars
                            - `Free y-scales`: enable separate `y` scales in each subplot
                            - `Log y-axis`: plot y-axes on a log-10 scale
                            - The grouping variables selected above will determine the
                            variable on the `x`-axis, the color variable, and faceting
                            variables used to create subplots.
                            
                            </div>
                            """),
                            custom_class="left-tip"
                        )),

                    ui.row(
                        ui.column(
                            8,
                            ui.row(
                                selectPred("yVar", "y:", 6, choices=yVar_exp),
                                ui.column(
                                    6,
                                    ui.input_radio_buttons(
                                        "plotGeom",
                                        ui.strong("Geometry:"),
                                        {"bars": "Bars",
                                         "lines": "Lines and Points",
                                         "points": "Points"}),
                                ),
                            ),

                            # TODO second y-axis
                            # ui.row(
                            #     ui.column(
                            #         6
                            #         ui.input_switch("plot2ndY",
                            #                         ui.strong("Second y-axis"),
                            #                     value = False)
                            #     ),
                            #     selectPred("yVar_2nd", "Second axis y:",
                            #                6, choices=yVar_exp,
                            #                selected="exposure")
                            # )
                        ),

                        ui.column(
                            4,
                            ui.input_switch("plotSmooth",
                                            ui.strong("Add smoothing"),
                                            value=False),
                            ui.input_switch("plotCI",
                                            ui.strong("Confidence intervals"),
                                            value=False),
                            ui.input_switch("plotFreeY",
                                            ui.strong("Free y-scales"),
                                            value=False),
                            ui.input_switch("plotLogY",
                                            ui.strong("Log y-axis"),
                                            value=False),
                        )
                    )

                ),

                ui.card(
                    ui.card_header(
                        ui.popover(
                            icon_svg("gear"),
                            ui.input_switch("plotResize", "Resize plot",
                                            value=False),
                            ui.input_slider("plotHeight", "Height (pixels):",
                                            200, 1500, value=500, step=50),
                            ui.input_slider("plotWidth", "Width (pixels):",
                                            200, 1500, value=1500, step=50)
                        )
                    ),
                    ui.output_ui("xpPlot"),
                    full_screen=True,
                    class_="no-overflow"
                )
            ),

            ui.nav_panel(
                "Table",
                ui.card(
                    ui.card_header(
                        ui.popover(
                            icon_svg("gear"),
                            ui.input_switch("tableCI",
                                            ui.strong("Confidence intervals"),
                                            value=False),
                            ui.input_switch("tableCredAdj",
                                            ui.strong(
                                                "Credibility-weighted termination rates"),
                                            value=False),
                            ui.input_switch("tableColorful",
                                            ui.strong("Include color scales"),
                                            value=True),
                            ui.input_slider("tableDecimals",
                                            ui.strong("Decimals:"),
                                            value=1, min=0, max=5),
                            ui.input_slider("tableFontsize",
                                            ui.strong("Font size multiple:"),
                                            value=100, min=50,
                                            max=150, step=5)
                        )
                    ),
                    gts.output_gt("xpTable"),
                    full_screen=True,
                    class_="no-overflow"
                )
            ),

            ui.nav_spacer(),
            ui.nav_menu(
                [icon_svg("download"), "Export"],
                ui.nav_panel(
                    ui.download_link("xpDownload", "Summary data (.csv)")
                ),
                ui.nav_panel(
                    ui.download_link("plotDownload", "Plot (.png)")
                ),
                # TODO - uncomment when great_tables adds a save method
                # ui.nav_panel(
                #     ui.download_link("tableDownload", "Table (.png)")
                # ),
                align='right'
            ),

            title="Output"
        ),

        ui.tags.style(ui.HTML("""
                              .html-fill-container > .html-fill-item {
                              overflow: visible; }
                              .html-fill-container > .no-overflow {
                              overflow: auto; }
                              """)),

        get_theme(bootswatch_theme) if bootswatch_theme is not None else "",
        title=title,
        fillable=False

    )

    def server(input, output, session):

        @reactive.Calc
        def yVar_exp2():
            choices = yVar_exp.copy()

            if has_expected:
                choices.extend(list(input.ex_checks()) +
                               [f"ae_{x}" for x in input.ex_checks()] +
                               [f"adj_{x}" for x in input.ex_checks()] +
                               ["All termination rates", "All A/E ratios"])

            return choices

        @reactive.Calc
        def yVar_trx2():
            if not has_trx:
                return []
            return (yVar_trx +
                    [f"pct_of_{x}_w_trx" for x in input.pct_checks()] +
                    [f"pct_of_{x}_all" for x in input.pct_checks()] +
                    list(input.pct_checks()) +
                    [f"{x}_w_trx" for x in input.pct_checks()])

        # update y variable selections in response to inputs
        @reactive.Effect
        @reactive.event(input.study_type,
                        (lambda: True) if not has_expected else input.ex_checks,
                        input.pct_checks)
        def _():
            if input.study_type() == "exp":
                choices = yVar_exp2()
            else:
                choices = yVar_trx2()

            ui.update_selectize(
                "yVar",
                choices=choices,
                selected=input.yVar() if input.yVar(
                ) in choices else choices[0]
            )

        # TODO second y-axis
        # @reactive.Effect
        # @reactive.event(input.study_type,
        #                 (lambda: True) if not has_expected else input.ex_checks,
        #                 input.pct_checks,
        #                 input.yVar)
        # def _():
        #     if input.study_type() == "exp":

        #         new_choices_2 = [
        #             x for x in yVar_exp() if
        #             x not in ["All termination rates", "All A/E ratios"]]

        #         if input.yVar() == "All termination rates":
        #             new_choices_2 = [
        #                 x for x in new_choices_2 if
        #                 x not in ['q_obs'] + input.ex_checks()]
        #         elif input.yVar() == "All A/E ratios":
        #             new_choices_2 = [
        #                 x for x in new_choices_2 if
        #                 x not in [f"ae_{x}" for x in input.ex_checks()]]

        #     else:
        #         new_choices_2 = yVar_trx2()

        #     ui.update_selectize(
        #         "yVar_2nd",
        #         choices = new_choices_2,
        #         selected = (input.yVar_2nd() if input.yVar_2nd() in
        #                     new_choices_2 else "exposure")
        #     )

        # disable color input when using special plots
        @reactive.Effect
        @reactive.event(input.yVar,
                        (lambda: True) if not has_expected else input.ex_checks)
        def _():
            if input.yVar() in ["All termination rates", "All A/E ratios"]:
                ui.update_selectize(
                    "colorVar", choices=["Series"], selected="Series")
            else:
                ui.update_selectize(
                    "colorVar",
                    choices=["None"] + preds_small,
                    selected=(input.colorVar() if input.colorVar() in
                              ["None"] + preds_small else "None")
                )

        # notification in pause mode
        @reactive.Effect
        def _():
            if not input.play():
                ui.notification_show("Reactivity is paused...",
                                     duration=None,
                                     id="paused",
                                     close_button=False)
            else:
                ui.notification_remove("paused")

        @reactive.Calc
        def active_filters():

            # TODO - update when validate() is added to Shiny for Python all
            # instances of `if not input.play()` can be removed.
            # ui.validate(ui.need(input.play(), "Paused"))
            if not input.play():
                return None

            def is_active(x):
                info = preds.loc[x]
                scope = info['scope']
                selected = input["i_" + x]()
                if info['dtype'] == "Dates":
                    res = (scope[0] != pd.to_datetime(selected[0]) or
                           scope[1] != pd.to_datetime(selected[1]))
                elif info['is_number']:
                    res = (not np.isclose(scope[0], selected[0]) or
                           not np.isclose(scope[1], selected[1]))
                else:
                    res = len(set(scope).difference(selected)) > 0
                if res:
                    return x

            keep = [is_active(x) for x in preds.index]

            return [x for x in keep if x is not None]

        # reactive data
        @reactive.Calc
        def rdat():

            if not input.play():
                return None

            # function to build filter expressions
            def expr_filter(x):

                inp_val = input["i_" + x]()

                # ensure that dates are quoted
                if preds.loc[x]['dtype'] == 'Dates':
                    inp_val = [f"'{a}'" for a in inp_val]

                # create filter expressions
                if preds.loc[x]['dtype'] in ['Dates', 'Numeric']:
                    res = f"({x} >= {inp_val[0]}) & ({x} <= {inp_val[1]})"
                else:
                    # categorical
                    res = f"({x}.isin({inp_val}))"

                return res

            # determine which filters are active
            filters = [expr_filter(x) for x in active_filters()]

            # if no active filters, return the current data
            if len(filters) == 0:
                return expo

            # apply filters
            filters = ' & '.join(filters)
            new_expo = deepcopy(expo)
            new_expo.data = new_expo.data.query(filters)

            return new_expo

        # experience study
        @reactive.Calc
        def rxp():

            if not input.play():
                return None

            # TODO - add when validate is added to shiny for python
            # ui.validate(ui.need(rdat().data is not None,
            #                     "No data remaining after applying filters."))
            # temporary
            if rdat().data.shape[0] == 0:
                return None

            expo = rdat()

            groups = [input.xVar(), input.colorVar()] + \
                list(input.facetVar())
            groups = [x for x in groups if x not in ["None", "Series"]]
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
                                  credibility=credibility,
                                  expected=ex,
                                  conf_level=conf_level,
                                  cred_r=cred_r,
                                  conf_int=True))
            else:
                return (expo.
                        groupby(*groups).
                        trx_stats(percent_of=list(input.pct_checks()),
                                  trx_types=list(input.trx_types_checks()),
                                  combine_trx=input.trx_combine(),
                                  conf_int=True,
                                  conf_level=conf_level))

        @reactive.Calc
        def rplot():

            if not input.play():
                return None
            if rdat().data.shape[0] == 0:
                return None

            if ((input.study_type() == "exp") & (input.yVar() in yVar_trx2()) &
                    (input.yVar() != "exposure")):
                return None
            if ((input.study_type() == "trx") & (input.yVar() in yVar_exp2()) &
                    (input.yVar() != "exposure")):
                return None

            new_rxp = deepcopy(rxp())

            if input.xVar() != "None":
                x = input.xVar()
            else:
                new_rxp.data["All"] = ""
                x = "All"

            if input.colorVar() != "None":
                color = input.colorVar()
            else:
                color = None

            # set up y-variable and plotting function
            if input.yVar() == "All termination rates":
                y = "Rate"
                plot_fun = new_rxp.plot_termination_rates
            elif input.yVar() == "All A/E ratios":
                y = 'A/E ratio'
                plot_fun = new_rxp.plot_actual_to_expected
            else:
                if color is not None:
                    if not {input.yVar(), color}.issubset(new_rxp.data.columns):
                        return None
                else:
                    if input.yVar() not in new_rxp.data.columns:
                        return None
                y = input.yVar()
                plot_fun = new_rxp.plot

            # TODO second axis
            # second_y = input.yVar_2nd()

            if color is not None:
                color = input.colorVar()
                mapping = aes(x, y, color=color,
                              fill=color, group=color)
            else:
                mapping = aes(x, y)

            # y labels
            # TODO a function is needed here for second axis labels in the future
            y_comma = ["claims", "n_claims", "exposure",
                       "trx_n", "trx_flag", "trx_amt",
                       "avg_trx", "avg_all"]
            if has_trx:
                y_comma = (y_comma + list(input.pct_checks()) +
                           [i + "_w_trx" for i in input.pct_checks()])
            if input.yVar() in y_comma:
                def y_labels(l): return [f"{v:,.0f}" for v in l]
            elif input.yVar() == "trx_freq":
                def y_labels(l): return [f"{v:,.1f}" for v in l]
            else:
                def y_labels(l): return [f"{v * 100:.1f}%" for v in l]

            if len(input.facetVar()) == 0:
                p = plot_fun(mapping=mapping, geoms=input.plotGeom(),
                             y_labels=y_labels,
                             y_log10=input.plotLogY(),
                             conf_int_bars=input.plotCI())
            else:
                facets = list(input.facetVar())

                p = plot_fun(mapping=mapping,
                             geoms=input.plotGeom(),
                             y_labels=y_labels,
                             facets=facets,
                             scales="free_y" if input.plotFreeY() else "fixed",
                             y_log10=input.plotLogY(),
                             conf_int_bars=input.plotCI())

            if input.plotSmooth():
                p = p + geom_smooth(method="loess")

            return (p +
                    theme_light() +
                    theme(axis_text=element_text(size=10),
                          strip_text=element_text(size=12),
                          strip_background=element_rect(fill="#43536b")
                          )
                    )

        @output
        @render.plot
        def oplot():
            return rplot()

        @output
        @render.ui()
        def xpPlot():
            return ui.output_plot(
                "oplot",
                height=input.plotHeight() if input.plotResize() else "500px",
                width=input.plotWidth() if input.plotResize() else None)

        @reactive.Calc
        def rtable():
            if not input.play():
                return None
            if rdat().data.shape[0] == 0:
                return None

            if (input.study_type() == "exp"):
                return rxp().table(
                    show_conf_int=input.tableCI(),
                    show_cred_adj=input.tableCredAdj(),
                    colorful=input.tableColorful(),
                    decimals=input.tableDecimals(),
                    fontsize=input.tableFontsize())
            else:
                return rxp().table(
                    show_conf_int=input.tableCI(),
                    colorful=input.tableColorful(),
                    decimals=input.tableDecimals(),
                    fontsize=input.tableFontsize())

        @output
        @gts.render_gt()
        def xpTable():
            return rtable()

        @output
        @render.text()
        def rem_pct():
            return f"{(rdat().data.shape[0] / total_rows) * 100:.0f}%"

        @output
        @render.text()
        def rem_rows():
            return f"Remaining rows: {rdat().data.shape[0]:,d}"

        @output
        @render.plot(pad_inches=0)
        def filter_pie():
            return plt.pie([rdat().data.shape[0],
                            total_rows - rdat().data.shape[0]],
                           wedgeprops=dict(width=0.5), startangle=90,
                           colors=["#033C73", "#BBBBBB"])

        # function to describe filters in words
        def describe_filter(x):

            selected = input["i_" + x]()
            info = preds.loc[x]
            choices = info['scope']

            # numeric or date
            if (info['dtype'] == "Numeric") | (info['dtype'] == "Dates"):

                if (info['dtype'] == 'Dates'):
                    selected = pd.to_datetime(selected)

                if selected[0] == selected[1]:
                    # exactly equal
                    return f"{x} == {selected[0]}"
                elif selected[0] > choices[0]:
                    if selected[1] < choices[1]:
                        # between
                        return f"{selected[0]} <= {x} <= {selected[1]}"
                    else:
                        # strictly greater than
                        return f"{x} >= {selected[0]}"
                else:
                    # strictly less than
                    return f"{x} <= {selected[1]}"

            elif len(selected) == 0:
                # categorical
                # nothing selected
                return f"{x} is nothing"
            else:
                def or_list(y):
                    y = list(y)
                    if len(y) > 2:
                        y[len(y) - 1] = "or " + y[len(y) - 1]
                        return ", ".join(y)
                    elif len(y) == 1:
                        return str(y[0])
                    else:
                        return f"{y[0]} or {y[1]}"

                if len(selected) < 0.5 * info.n_unique:
                    # minority selected - list all selections
                    return f"{x} is {or_list(selected)}"
                else:
                    # majority selected - list all NOT selected
                    y = set(choices).difference(selected)
                    return f"{x} is NOT {or_list(y)}"

        # filter description output
        @output
        @render.text()
        def filter_desc():
            return "\n".join([describe_filter(x) for x in active_filters()])

        @output
        @render.text()
        def filter_desc_header():
            if len(active_filters()) > 0:
                return "Active filters"

        # exporting

        # function factory for output file names
        def export_path(study_type, x, extension):
            # TODO should this use tempfile.TemporaryDirectory?
            return f"{study_type}-{x}-{datetime.today().isoformat(timespec='minutes')[:10]}.{extension}"

        @render.download(
            filename=lambda: export_path(input.study_type(), "data", "csv")
        )
        def xpDownload():
            if not input.play():
                return None
            if rdat().data.shape[0] == 0:
                return None
            with io.BytesIO() as buf:
                rxp().data.to_csv(buf)
                yield buf.getvalue()

        @render.download(
            filename=lambda: export_path(input.study_type(), "plot", "png")
        )
        def plotDownload():
            if not input.play():
                return None
            if rdat().data.shape[0] == 0:
                return None
            with io.BytesIO() as buf:
                # note the conversion of height / width to pixels below
                rplot().save(buf,
                             height=input.plotHeight() if input.plotResize() / 96
                             else None,
                             width=input.plotWidth() if input.plotResize() / 96
                             else None,
                             units='in')
                yield buf.getvalue()

        # TODO - uncomment when great_tables adds a save method
        # @render.download(
        #     filename=lambda: export_path(input.study_type(), "table", "png")
        # )
        # def tableDownload():
        #     if not input.play():
        #         return None
        #     if rdat().data.shape[0] == 0:
        #         return None
        #     with io.BytesIO() as buf:
        #         rtable().save(buf)
        #         yield buf.getvalue()

    # Run the application
    return App(app_ui, server)
