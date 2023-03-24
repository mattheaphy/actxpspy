import pkg_resources
import pandas as pd

def load_toy_census() -> pd.DataFrame:
    """
    # Toy policy census data
    
    A tiny dataset containing 3 policies: one active, one terminated due to
    death, and one terminated due to surrender.    

    ## Returns:
    
    A data frame with 3 rows and 4 columns:
    
    - `pol_num` = policy number
    - `status` = policy status
    - `issue_date` = issue date
    - `term_date` = termination date
    
    """
    stream = pkg_resources.resource_stream(__name__, 'data/toy_census.csv')
    return pd.read_csv(stream,
                       index_col=0,
                       dtype={'pol_num': int,
                              'status': 'category'},
                       parse_dates=['issue_date', 'term_date'])