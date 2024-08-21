from ..run_glm import *
import pytest
import pandas as pd
import numpy as np

# define some fixtures
@pytest.fixture
def confounds_df():
    df = pd.DataFrame(data={"framewise_displacement": np.random.rand(50), 
                            "global_signal": np.random.rand(50) * 5000,
                            "volterra": np.ones(50)}) 
    return df


@pytest.fixture
def confounds_columns(confounds_df):
    cols = [col for col in confounds_df.columns if "volterra" not in col]
    return cols


def test_make_option():
    assert make_option(True) == " "
    assert make_option(True, key="myopt") == "--myopt "
    assert make_option(['1','2','3']) == " 1 2 3"
    assert make_option("hello") == " hello"
    assert make_option("hello", key="myopt") == "--myopt hello"


@pytest.mark.parametrize("demean,linear_trend,spike_threshold,volterra_expansion,volterra_columns",
                         [(False, False, None, None, None),
                          (True, False, None, None, None),
                          (False, True, None, None, None),
                          (True, True, None, None, None),
                          (True, True, 0.5, None, None),
                          (True, True, 0.5, 1, ["volterra"]),
                          (True, True, 0.5, 3, ["volterra"])])
def test_make_noise_ts(tmp_path,
                       confounds_df,
                       confounds_columns,
                       demean,
                       linear_trend,
                       spike_threshold,
                       volterra_expansion,
                       volterra_columns):
    confounds_file = tmp_path / "tmp.tsv"
    confounds_df.to_csv(confounds_file, sep='\t')
    noise_ts = make_noise_ts(confounds_file,
                       confounds_columns,
                       demean,
                       linear_trend,
                       spike_threshold,
                       volterra_expansion,
                       volterra_columns)
    if spike_threshold:
        assert spike_threshold > 0
        spike_column_num = (noise_ts.loc[:, "framewise_displacement"] > spike_threshold).sum()
        assert len([col for col in noise_ts.columns if "spike" in col]) == spike_column_num
       
    assert "volterra_expansion" in locals() and "volterra_columns" in locals()
    if volterra_expansion:
        for i in range(volterra_expansion):
            # Check that each volterra column has been shifted appropriately
            assert (noise_ts.loc[0:i, f"volterra_{i+1}"] == 0).all()

