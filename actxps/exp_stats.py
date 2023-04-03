import pandas as pd
import numpy as np
from scipy.stats import norm
from warnings import warn
from actxps.expose import ExposedDF


class ExpStats():
    """
    TODO
    """

    def __init__(self,
                 expo: ExposedDF,
                 target_status: str | list | np.ndarray = None,
                 expected: str | list | np.ndarray = None,
                 credibility: bool = False,
                 cred_p: float = 0.95,
                 cred_r: float = 0.05,
                 wt: str = None):
        
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

        res = expo.data.assign(
            n_claims=expo.data.status.isin(target_status)
        )

        # set up weights
        if wt is not None:

            assert isinstance(wt, str), "`wt` must have type `str`"
            res = res.rename(columns={wt: 'weight'})
            res['claims'] = res.n_claims * res.weight
            res['exposure'] = res.exposure * res.weight
            res['weight_sq'] = res.weight ** 2
            res['weight_n'] = 1

        else:
            res['claims'] = res.n_claims

        # finish exp stats

        if expo.groups is not None:
            res = (res.groupby(expo.groups).
                   apply(ExpStats._finish_exp_stats,
                         expected=expected,
                         credibility=credibility,
                         cred_p=cred_p,
                         cred_r=cred_r, wt=wt).
                   reset_index().
                   drop(columns=[f'level_{len(expo.groups)}']))

        else:
            res = ExpStats._finish_exp_stats(res,
                                             expected, 
                                             credibility,
                                             cred_p,
                                             cred_r,
                                             wt)


        self.data = res
        self.groups = expo.groups
        self.target_status = target_status
        self.end_date = expo.end_date
        self.start_date = expo.start_date
        self.expected = expected
        self.wt = wt
        self.exp_params = {'credibility': credibility,
                           'cred_p': cred_p,
                           'cred_r': cred_r}
        
        return None

    @staticmethod
    def _finish_exp_stats(data: pd.DataFrame, 
                          expected: str | list | np.ndarray, 
                          credibility: bool, 
                          cred_p: float, 
                          cred_r: float, 
                          wt: str):
        """
        Internal method for finalizing experience study summaries.
        """
        # dictionary of summarized values
        fields = {'n_claims': sum(data.n_claims),
                  'claims': sum(data.claims),
                  'exposure': sum(data.exposure)}

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
                    'credibility':
                        np.minimum(1, lambda x:
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

        cols = (['n_claims', 'claims', 'exposure', 'q_obs'] +
                [k for k in expected] +
                ['ae_' + k for k in expected] +
                (['credibility'] if credibility else None) +
                (['adj_' + k for k in expected] if credibility else None) +
                (['weight', 'weight_sq', 'weight_n']
                if wt is not None else None)
                )

        data = data[cols]

        return data