import pandas as pd
import numpy as np
from scipy.stats import norm
from warnings import warn
from functools import singledispatchmethod
from actxps.expose import ExposedDF


class ExpStats():
    """
    TODO
    """

    @singledispatchmethod
    def __init__(self,
                 expo: ExposedDF,
                 target_status: str | list | np.ndarray = None,
                 expected: str | list | np.ndarray = None,
                 credibility: bool = False,
                 cred_p: float = 0.95,
                 cred_r: float = 0.05,
                 wt: str = None):

        self.data = None
        # set up target statuses. First, attempt to use the statuses that
        # were passed. If none, then use the target_status
        # property from the ExposedDF. If that property is None, assume
        # all statuses but the first are the target.
        if target_status is None:
            target_status = expo.target_status
        if target_status == [None]:
            target_status = list(expo.data.status.cat.categories[1:])
            warn(f"No target status was provided. {', '.join(target_status)} "
                 "was assumed.")

        data = expo.data.assign(
            n_claims=expo.data.status.isin(target_status)
        )

        # set up weights
        if wt is not None:

            assert isinstance(wt, str), "`wt` must have type `str`"
            data = data.rename(columns={wt: 'weight'})
            data['claims'] = data.n_claims * data.weight
            data['exposure'] = data.exposure * data.weight
            data['weight_sq'] = data.weight ** 2
            data['weight_n'] = 1

        else:
            data['claims'] = data.n_claims

        cred_params = {'credibility': credibility,
                       'cred_p': cred_p,
                       'cred_r': cred_r}

        # set up properties and summarize data
        self._finalize(data, expo.groups, target_status,
                       expo.end_date, expo.start_date,
                       expected, wt, cred_params)

        return None

    def _finalize(self,
                  data: pd.DataFrame,
                  groups,
                  target_status,
                  end_date,
                  start_date,
                  expected,
                  wt,
                  cred_params):
        """
        Internal method for finalizing experience study summary objects
        """

        # set up properties
        self.groups = groups
        self.target_status = target_status
        self.end_date = end_date
        self.start_date = start_date
        self.expected = expected
        self.wt = wt
        self.cred_params = cred_params

        # finish exp stats
        if groups is not None:
            res = (data.groupby(groups).
                   apply(self._calc).
                   reset_index().
                   drop(columns=[f'level_{len(groups)}']))

        else:
            res = self._calc(data)

        self.data = res

        return None

    def _calc(self, data: pd.DataFrame):
        """
        Support function for summarizing data for one group
        """
        # dictionary of summarized values
        fields = {'n_claims': sum(data.n_claims),
                  'claims': sum(data.claims),
                  'exposure': sum(data.exposure)}

        expected = self.expected
        wt = self.wt
        credibility = self.cred_params['credibility']
        cred_p = self.cred_params['cred_p']
        cred_r = self.cred_params['cred_r']

        if expected is not None:

            if isinstance(expected, str | int):
                expected = [expected]

            ex_mean = {k: np.average(data[k], weights=data.exposure)
                       for k in expected}
            fields.update(ex_mean)

        # additional columns for weighted studies
        if wt is not None:
            wt_forms = {
                'weight': sum(data.weight),
                'weight_sq': sum(data.weight_sq),
                'weight_n': sum(data.weight_n)}
            fields.update(wt_forms)

            wt_forms2 = {
                'ex_wt': lambda x: x.weight / x.weight_n,
                'ex2_wt': lambda x: x.weight_sq / x.weight_n}
        else:
            wt_forms2 = {}

        # credibility formulas - varying by weights
        if credibility:
            y = (norm.ppf((1 + cred_p) / 2) / cred_r) ** 2

            if wt is None:
                cred = {
                    'credibility': lambda x:
                        np.minimum(1,
                                   (x.n_claims / (y * (1 - x.q_obs)))
                                   ** 0.5)
                }
            else:
                cred = {
                    'credibility': lambda x:
                        np.minimum(1,
                                   (x.n_claims / (
                                       y * ((x.ex2_wt - x.ex_wt ** 2) *
                                            x.weight_n / (x.weight_n - 1) /
                                            x.ex_wt ** 2 + 1 - x.q_obs)))
                                   ** 0.5)
                }

        else:
            cred = {}

        # dictionary of columns that depend on summarized values
        fields2 = {
            'q_obs': lambda x: x.claims / x.exposure
        }
        fields2.update(wt_forms2)
        fields2.update(cred)

        # convert dataults to a data frame
        data = pd.DataFrame(fields, index=range(1)).assign(**fields2)

        # add A/E's and adjusted q's
        if expected is not None:
            for k in expected:
                data['ae_' + k] = data.q_obs / data[k]

        if credibility & (expected is not None):
            for k in expected:
                data['adj_' + k] = (data.credibility * data.q_obs +
                                    (1 - data.credibility) * data[k])

        # rearrange and drop columns
        if wt is not None:
            data = data.drop(columns=['ex_wt', 'ex2_wt'])

        cols = ['n_claims', 'claims', 'exposure', 'q_obs']
        if expected is not None:
            cols.extend([k for k in expected] +
                        ['ae_' + k for k in expected])

        if credibility:
            cols.extend(['credibility'])
            if expected:
                cols.extend(['adj_' + k for k in expected])

        if wt is not None:
            cols.extend(['weight', 'weight_sq', 'weight_n'])

        data = data[cols]

        return data

    def summary(self, *by):
        """
        TODO
        """

        by = list(by)

        if len(by) == 0:
            by = None
        else:
            assert all(pd.Series(by).isin(self.data.columns)), \
                "All grouping variables passed to `*by` must be in the`.data` property."

        self.groups = by

        return ExpStats('from_summary', self)

    @ __init__.register(str)
    def _special_init(self,
                      style: str,
                      old_self):
        """
        Special constructor for the ExpStats class. This constructor is used
        by the `summary()` class method to create new summarized instances.
        """

        assert style == "from_summary"
        self.data = None
        self._finalize(old_self.data, old_self.groups, old_self.target_status,
                       old_self.end_date, old_self.start_date,
                       old_self.expected, old_self.wt, old_self.cred_params)

    def __repr__(self):
        repr = "Experience study results\n\n"

        if self.groups is not None:
            repr += f"Groups: {', '.join([str(i) for i in self.groups])}\n"

        repr += f"Target status: {', '.join([str(i) for i in self.target_status])}\n"
        f"Study range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n"

        if self.expected is not None:
            repr += f"Expected values: {', '.join([str(i) for i in self.expected])}\n"

        if self.wt is not None:
            repr += f"Weighted by: {self.wt}\n"

        if self.data is not None:
            repr = (repr +
                    f"\n\nA DataFrame: {self.data.shape[0]:,} x {self.data.shape[1]:,}" +
                    f'\n{self.data.head(10)}')

        return repr
