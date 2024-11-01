import pandas as pd
import numpy as np
import re

def _read_design_matrix(path: str,
                        sep: str = None,
                        dtype: type = None) -> np.typing.ArrayLike:
    if sep is None and re.match(r'^.*\.csv$', path):
        sep = ','
    elif sep is None and re.match(r'^.*\.tsv$', path):
        sep = '\t'
    if sep is None:
        raise ValueError("Unrecognized separator in design matrix file (should either be a .csv or .tsv file)")
    df = pd.DataFrame(path, sep=sep)
    return df.to_numpy()





