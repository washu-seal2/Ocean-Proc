#!/usr/bin/env python3
import numpy as np
import sys
from pathlib import Path
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import nibabel as nib
# from nilearn.glm.first_level import FirstLevelModel
# from nilearn.plotting import plot_design_matrix
# import matplotlib.pyplot as plt
import nilearn.masking as nmask
from nilearn.signal import butterworth, _handle_scrubbed_volumes
from scipy import signal
from scipy.stats import gamma
from ..oceanparse import OceanParser
from ..events_long import make_events_long
from ..utils import exit_program_early, add_file_handler, default_log_format, export_args_to_file, flags, debug_logging, log_linebreak, load_data
import logging
import datetime
from textwrap import dedent

"""
TODO: 
    * Find good way to pass hrf peak and undershoot variables
    * Save final noise df for each run
    * Debug Mode - save intermediate outputs
    * Options for highpass and lowpass filters
    * Function documentation and testing

"""
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(stream=sys.stdout)],
                    format=default_log_format)
logger = logging.getLogger()


@debug_logging
def load_data(func_file: str|Path,
              brain_mask: str = None,
              auto_mask: bool = False,
              need_tr: bool = False) -> np.ndarray:
    tr = None
    func_file = str(func_file)
    if need_tr:
        sidecar_file = func_file.split(".")[0] + ".json"
        assert os.path.isfile(sidecar_file), f"Cannot find the .json sidecar file for bold run: {func_file}"
        with open(sidecar_file, "r") as f:
            jd = json.load(f)
            tr = jd["RepetitionTime"]

    if func_file.endswith(".dtseries.nii") or func_file.endswith(".dscalar.nii"):
        img = nib.load(func_file)
        return (img.get_fdata(), tr, img.header)
    elif func_file.endswith(".nii") or func_file.endswith(".nii.gz"):
        if brain_mask:
            return (nmask.apply_mask(func_file, brain_mask), tr, None)
        elif auto_mask:
            mask_path = func_file.replace("desc-preproc_bold", "desc-brain_mask")
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(dedent(f"""
                                               --auto_mask flag was set, but cannot find an accompanying brain mask for the given BOLD run.
                                               BOLD path: {func_file}
                                               """))
            return (nmask.apply_mask(func_file, mask_path), tr, None)
        else:
            raise Exception("Volumetric data must also have an accompanying brain mask")
            # return None 


@debug_logging
def create_image(data: npt.ArrayLike,
                 brain_mask: str = None,
                 tr: float = None,
                 header: nib.cifti2.cifti2.Cifti2Header = None):
    img = None
    suffix = ".nii"
    d32k = 32492
    if brain_mask:
        img = nmask.unmask(data, brain_mask)
    else:
        ax0 = None
        if data.shape[0] > 1 and tr:
            ax0 = nib.cifti2.cifti2_axes.SeriesAxis(
                start=0.0,
                step=tr,
                size=data.shape[0]
            )
            suffix = ".dtseries" + suffix
        elif data.shape[0] == 1:
            ax0 = nib.cifti2.cifti2_axes.ScalarAxis(
                name=["beta"]
            )
            suffix = ".dscalar" + suffix
        else:
            raise RuntimeError("TR not supplied or data shape is incorrect")
        ax1 = None
        if header:
            ax1 = header.get_axis(1)
        else:
            # Need to change this default behavior to create a correct Brain Model axis
            ax1 = nib.cifti2.cifti2_axes.BrainModelAxis(
                name=(['CIFTI_STRUCTURE_CORTEX_LEFT']*d32k)+(['CIFTI_STRUCTURE_CORTEX_RIGHT']*d32k),
                vertex=np.concatenate((np.arange(d32k), np.arange(d32k))),
                nvertices={'CIFTI_STRUCTURE_CORTEX_LEFT':d32k, 'CIFTI_STRUCTURE_CORTEX_RIGHT':d32k}
            )
        img = nib.cifti2.cifti2.Cifti2Image(data, (ax0, ax1))
    return (img ,suffix)


def demean_detrend(func_data: npt.ArrayLike) -> np.ndarray:
    """
    Subtracts the mean and a least-squares-fit line from each timepoint at every vertex/voxel.
    
    Parameters
    ----------

    func_data: npt.ArrayLike 
        array containing functional timeseries data

    Returns
    -------
    data_dd: np.ndarray
        A demeaned/detrended copy of the input array
    """
    data_dd = signal.detrend(func_data, axis=0, type = 'linear')
    return data_dd


def create_hrf(time, time_to_peak=5, undershoot_dur=12):
    """
    This function creates a hemodynamic response function timeseries.

    Parameters
    ----------
    time: numpy array
        a 1D numpy array that makes up the x-axis (time) of our HRF in seconds
    time_to_peak: int
        Time to HRF peak in seconds. Default is 5 seconds.
    undershoot_dur: int
        Duration of the post-peak undershoot. Default is 12 seconds.

    Returns
    -------
    hrf_timeseries: numpy array
        The y-values for the HRF at each time point
    """

    peak = gamma.pdf(time, time_to_peak)
    undershoot = gamma.pdf(time, undershoot_dur)
    hrf_timeseries = peak - 0.35 * undershoot
    return hrf_timeseries


@debug_logging
def hrf_convolve_features(features: pd.DataFrame,
                          column_names: list = None,
                          time_col: str = 'index',
                          units: str = 's',
                          time_to_peak: int = 5,
                          undershoot_dur: int = 12):
    """
    This function convolves a hemodynamic response function with each column in a timeseries dataframe.

    Parameters
    ----------
    features: DataFrame
        A Pandas dataframe with the feature signals to convolve.
    column_names: list
        List of columns names to use; if it is None, use all columns. Default is None.
    time_col: str
        The name of the time column to use if not the index. Default is "index".
    units: str
        Must be 'ms','s','m', or 'h' to denote milliseconds, seconds, minutes, or hours respectively.
    time_to_peak: int
        Time to peak for HRF model. Default is 5 seconds.
    undershoot_dur: int
        Undershoot duration for HRF model. Default is 12 seconds.

    Returns
    -------
    convolved_features: DataFrame
        The HRF-convolved feature timeseries
    """
    if not column_names:
        column_names = features.columns

    if time_col == 'index':
        time = features.index.to_numpy()
    else:
        time = features[time_col]
        features.index = time

    if units == 'm' or units == 'minutes':
        features.index = features.index * 60
        time = features.index.to_numpy()
    if units == 'h' or units == 'hours':
        features.index = features.index * 3600
        time = features.index.to_numpy()
    if units == 'ms' or units == 'milliseconds':
        features.index = features.index / 1000
        time = features.index.to_numpy()

    convolved_features = pd.DataFrame(index=time)
    hrf_sig = create_hrf(time, time_to_peak=time_to_peak, undershoot_dur=undershoot_dur)
    for a in column_names:
        convolved_features[a] = np.convolve(features[a], hrf_sig)[:len(time)]

    return convolved_features


def find_nearest(array, value):
    """
    Finds the smallest difference in 'value' and one of the 
    elements of 'array', and returns the index of the element

    :param array: a list of elements to compare value to
    :type array: a list or list-like object
    :param value: a value to compare to elements of array
    :type value: integer or float
    :return: integer index of array
    :rtype: int
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return(array[idx])


@debug_logging
def make_noise_ts(confounds_file: str,
                  confounds_columns: list,
                  demean: bool = False,
                  linear_trend: bool = False,
                  spike_threshold: float = None,
                  volterra_expansion: int = None,
                  volterra_columns: list = None):
    select_columns = set(confounds_columns)
    fd = "framewise_displacement"
    if volterra_columns:
        select_columns.update(volterra_columns)
    if spike_threshold:
        select_columns.add(fd)
    nuisance = pd.read_csv(confounds_file, delimiter='\t').loc[:,list(select_columns)]
    if fd in select_columns:
        nuisance.loc[0, fd] = 0

    if demean: 
        nuisance["mean"] = 1

    if linear_trend:
        nuisance["trend"] = np.arange(0, len(nuisance))

    """
    Add a new column denoting indices where a frame 
    is censored for each row in the framewise_displacement
    that is larger than spike_threshold.
    """
    if spike_threshold:
        b = 0
        for a in range(len(nuisance)):
            if nuisance.loc[a,fd] > spike_threshold:
                nuisance[f"spike{b}"] = 0
                nuisance.loc[a, f"spike{b}"] = 1
                b += 1
        if fd not in confounds_columns:
                nuisance.drop(columns=fd, inplace=True)

    if volterra_expansion and volterra_columns:
        for vc in volterra_columns:
            for lag in range(volterra_expansion):
                nuisance.loc[:, f"{vc}_{lag+1}"] = nuisance.loc[:, vc].shift(lag+1)
        nuisance.fillna(0, inplace=True)
    elif volterra_expansion:
        raise RuntimeError("You must specify which columns you'd like to apply Volterra expansion to.")
    elif volterra_columns:
        raise RuntimeError("You must specify the lag applied in Volterra expansion.")

    return nuisance


#TODO: maybe write a validator for the input task file?
@debug_logging
def events_to_design(events_long: pd.DataFrame,
                     fir: int = None,
                     hrf: tuple[int] = None,
                     fir_list: list[str] = None,
                     hrf_list: list[str] = None,
                     unmodeled_list: list[str] = None,
                     output_path: Path = None):
    """
    Builds an initial design matrix from an events long DataFrame. You can 
    convolve specified event types with an hemodynamic response function
    (HRF).

    The events are expected to be in a .tsv file, with the following columns:

    trial_type: a string representing the unique trial type. oceanfla gives you the 
        freedom to arrange these trial types in any way you want; they can represent 
        events on their own, combinations of concurrent events, etc. 
    onset: the onset time of an event
    duration: the duration of an event
    
    Parameters
    ----------
    events_long: pd.DataFrame
        A dataframe representing a long formatted events file
    fir: int = None
        An int denoting the order of an FIR filter
    hrf: tuple[int] = None
        A 2-length tuple, where hrf[0] denotes the time to the peak of an HRF, and hrf[1] denotes the duration of its "undershoot" after the peak.
    fir_list: list[str] = None
        A list of column names denoting which columns should have an FIR filter applied.
    hrf_list: list[str] = None
        A list of column names denoting which columns should be convolved with the HRF function defined in the hrf tuple.
    unmodeled_list: list[str] = None
        A list of column names denoting which columns should not be modeled by neither hrf or fir, but still included in the design matrix.
    output_path: pathlib.Path = None
        A path to a csv file where the created events matrix should be saved to.

    Returns
    -------
    (events_matrix, conditions): tuple
        A tuple containing the DataFrame with filtered/convolved columns and a list of unique trial names.
    """
    events_matrix = events_long.copy()
    if fir and fir <= 0:
        raise ValueError(f"fir value must be greater than 0. Current fir: {fir}")
    if hrf and not (len(hrf) == 2 and hrf[0] > 0 and hrf[1] > 0):
        raise ValueError(f"hrf tuple must contain two integers greater than 0. Current hrf tuple: {hrf}")
    # If both FIR and HRF are specified, we should have at least one list 
    # of columns for one of the categories specified.
    if (fir and hrf) and not (fir_list or hrf_list): 
        raise RuntimeError("Both FIR and HRF were specified, but you need to specify at least one list of variables (fir_list or hrf_list)")
    # fir_list and hrf_list must not have overlapping columns
    if (fir_list and hrf_list) and not set(fir_list).isdisjoint(hrf_list):
        raise RuntimeError("Both FIR and HRF lists of variables were specified, but they overlap.")
    conditions = [s for s in np.unique(events_matrix.columns)] # unique trial types
    residual_conditions = [c for c in conditions if c not in unmodeled_list] if unmodeled_list else conditions
    if (fir and hrf) and (bool(fir_list) ^ bool(hrf_list)): # Create other list if only one is specified
        if fir_list:
            hrf_list = [c for c in residual_conditions if c not in fir_list]
        elif hrf_list:
            fir_list = [c for c in residual_conditions if c not in hrf_list]
        assert set(hrf_list).isdisjoint(fir_list)
        
    if fir:
        fir_conditions = residual_conditions
        if fir_list and len(fir_list) > 0:
            fir_conditions = [c for c in residual_conditions if c in fir_list]
        residual_conditions = [c for c in residual_conditions if c not in fir_conditions]
        
        col_names = {c:c+"_00" for c in fir_conditions}
        events_matrix = events_matrix.rename(columns=col_names)
        fir_cols_to_add = dict()
        for c in fir_conditions:
            for i in range(1, fir):
                fir_cols_to_add[f"{c}_{i:02d}"] = np.array(np.roll(events_matrix.loc[:,col_names[c]], shift=i, axis=0))
                # so events do not roll back around to the beginnin
                fir_cols_to_add[f"{c}_{i:02d}"][:i] = 0
        events_matrix = pd.concat([events_matrix, pd.DataFrame(fir_cols_to_add, index=events_matrix.index)], axis=1)
        events_matrix = events_matrix.astype(int)
    if hrf:
        hrf_conditions = residual_conditions
        if hrf_list and len(hrf_list) > 0:
            hrf_conditions = [c for c in residual_conditions if c in hrf_list]
        residual_conditions = [c for c in residual_conditions if c not in hrf_conditions]
        
        cfeats = hrf_convolve_features(features=events_matrix, 
                                       column_names=hrf_conditions,
                                       time_to_peak=hrf[0],
                                       undershoot_dur=hrf[1])
        for c in hrf_conditions:
            events_matrix[c] = cfeats[c]
    
    if len(residual_conditions) > 0 and logger:
        logger.warning(dedent(f"""The following trial types were not selected under either of the specified models
                           and were also not selected to be left unmodeled. These variables will not be included in the design matrix:\n\t {residual_conditions}"""))
        events_matrix = events_matrix.drop(columns=residual_conditions)

    if output_path:
        logger.debug(f" saving events matrix to file: {output_path}")
        events_matrix.to_csv(output_path)
    
    return (events_matrix, conditions)


@debug_logging
def filter_data(func_data: npt.ArrayLike,
                mask: npt.ArrayLike,
                tr: float,
                low_pass: float = 0.1,
                high_pass: float = 0.008):
    
    """
    Apply a lowpass, highpass, or bandpass filter to the functional data. Masked frames
    are interploated using a cubic splice function before filtering. The returned array 
    is the functional data after interpolation and filtering.

    Parameters
    ----------
    func_data: npt.ArrayLike
        A numpy array representing BOLD data
    mask: npt.ArrayLike
        A numpy array representing a mask along the first axis (time axis) of the BOLD data
    tr: float
        Repetition time at the scanner
    high_cut: float
        Frequency above which the bandpass filter will be applied
    low_cut: float
        Frequency below which the bandpass filter will be applied

    Returns
    -------

    filtered_data: npt.ArrayLike
        A numpy array representing BOLD data with the filter applied
    """
    assert mask.shape[0] == func_data.shape[0], "the mask must be the same length as the functional data"
    assert mask.dtype == bool

    # if the mask is excluding frames, interpolate the censored frames
    if np.sum(mask) < mask.shape[0]:
        func_data, _, mask = _handle_scrubbed_volumes(
            signals=func_data,
            confounds=None,
            sample_mask=mask,
            filter_type="butterworth",
            t_r=tr,
            extrapolate=True
        )

    filtered_data = butterworth(
        signals=func_data,
        sampling_rate=1.0 / tr,
        low_pass=low_pass,
        high_pass=high_pass
    )

    return filtered_data


@debug_logging
def create_final_design(data_list: list[npt.ArrayLike],
                        design_list: list[tuple[pd.DataFrame, int]],
                        noise_list: list[pd.DataFrame] = None,
                        exclude_global_mean: bool = False):
    """
    Creates a final, concatenated design matrix for all functional runs in a session

    Parameters
    ----------

    data_list: list[npt.ArrayLike]
        List of numpy arrays representing BOLD data
    design_list: list[pd.DataFrame]
        List of created design matrices corresponding to each respective BOLD run in data_list
    noise_list: list[pd.DataFrame]
        List of DataFrame objects corresponding to models of noise for each respective BOLD run in data_list
    exclude_global_mean: bool
        Flag to indicate that a global mean should not be included into the final design matrix

    Returns
    -------

    Returns a tuple containing the final concatenated data in index 0, and the 
    final concatenated design matrix in index 1.
    """
    num_runs = len(data_list)
    assert num_runs == len(design_list), "There should be the same number of design matrices and functional runs"
    
    design_df_list = [t[0] for t in design_list]
    if noise_list:
        assert num_runs == len(noise_list), "There should be the same number of noise matrices and functional runs"
        for i in range(num_runs):
            noise_df = noise_list[i]
            assert len(noise_df) == len(design_df_list[i])
            run_num = design_list[i][1]
            rename_dict = dict()
            for c in noise_df.columns:
                if ("trend" in c) or ("mean" in c) or ("spike" in c):
                    rename_dict[c] = f"run-{run_num:02d}_{c}"
            noise_df = noise_df.rename(columns=rename_dict)
            noise_list[i] = noise_df
            design_df_list[i] = pd.concat([design_df_list[i].reset_index(drop=True), noise_df.reset_index(drop=True)], axis=1)

    final_design = pd.concat(design_df_list, axis=0, ignore_index=True).fillna(0)
    if not exclude_global_mean:
        final_design.loc[:, "global_mean"] = 1
    final_data = np.concat(data_list, axis=0)
    return (final_data, final_design)


@debug_logging
def massuni_linGLM(func_data: npt.ArrayLike,
                   design_matrix: pd.DataFrame,
                   mask: npt.ArrayLike):
    """
    Compute the mass univariate GLM.

    Parameters
    ----------

    func_data: npt.ArrayLike
        Numpy array representing BOLD data
    design_matrix: pd.DataFrame
        DataFrame representing a design matrix for the GLM
    mask: npt.ArrayLike
        Numpy array representing a mask to apply to the two other parameters
    """
    assert mask.shape[0] == func_data.shape[0], "the mask must be the same length as the functional data"
    assert mask.dtype == bool

    # apply the mask to the data
    design_matrix = design_matrix.to_numpy()
    masked_func_data = func_data[mask, :]
    masked_design_matrix = design_matrix[mask, :]

    ss = StandardScaler()

    # standardize the masked data
    masked_func_data = ss.fit_transform(masked_func_data)
    masked_design_matrix = ss.fit_transform(masked_design_matrix)

    # comput beta values
    inv_mat = np.linalg.pinv(masked_design_matrix)
    beta_data = np.dot(inv_mat, masked_func_data)

    # standardize the unmasked data
    func_data = ss.fit_transform(func_data)
    design_matrix = ss.fit_transform(design_matrix)

    # compute the residuals with unmasked data
    est_values = np.dot(design_matrix, beta_data)
    resids = func_data - est_values

    return (beta_data, resids)



def main():
    parser = OceanParser(
        prog="oceanfla",
        description="Ocean Labs first level analysis",
        fromfile_prefix_chars="@",
        epilog="An arguments file can be accepted with @FILEPATH"
    )
    session_arguments = parser.add_argument_group("Session Specific")
    config_arguments = parser.add_argument_group("Configuration Arguments", "These arguments are saved to a file if the '--export_args' option is used")

    session_arguments.add_argument("--subject", "-su", required=True,
                        help="The subject ID")
    session_arguments.add_argument("--session", "-se", required=True,
                        help="The session ID")
    session_arguments.add_argument("--events_long", "-el", type=Path, nargs="?", const=lambda a: a.derivs_dir / a.preproc_subfolder,
                        help="""Path to the directory containing long formatted event files to use. 
                        Default is the derivatives directory containing preprocessed outputs.""")
    session_arguments.add_argument("--export_args", "-ea", type=Path,
                        help="Path to a file to save the current arguments.")
    session_arguments.add_argument("--debug", action="store_true",
                        help="Use this flag to save intermediate outputs for a chance to debug inputs")
    
    config_arguments.add_argument("--task", "-t", required=True,
                        help="The name of the task to analyze.")
    config_arguments.add_argument("--bold_file_type", "-ft", required=True,
                        help="The file type of the functional runs to use.")

    mask_arguments = config_arguments.add_mutually_exclusive_group()
    mask_arguments.add_argument("--brain_mask", "-bm", type=Path,
                        help="If the bold file type is volumetric data, a brain mask must also be supplied.")
    mask_arguments.add_argument("--auto_mask", "-am", type=Path,
                        help="If the bold file type is volumetric data, try to automatically assign fMRIprep/Nibabies-generated brain masks to each run.",
                        action='store_true')

    config_arguments.add_argument("--derivs_dir", "-d", type=Path, required=True,
                        help="Path to the BIDS formatted derivatives directory containing processed outputs.")
    config_arguments.add_argument("--preproc_subfolder", "-pd", type=str, default="fmriprep",
                        help="Name of the subfolder in the derivatives directory containing the preprocessed bold data")
    config_arguments.add_argument("--raw_bids", "-r", type=Path, required=True,
                        help="Path to the BIDS formatted raw data directory for this dataset.")
    config_arguments.add_argument("--derivs_subfolder", "-ds", default="first_level",
                        help="The name of the subfolder in the derivatives directory where bids style outputs should be stored. The default is 'first_level'.")
    config_arguments.add_argument("--output_dir", "-o", type=Path,
                        help="Alternate Path to a directory to store the results of this analysis. Default is '[derivs_dir]/first_level/'")
    config_arguments.add_argument("--fir", "-ff", type=int,
                        help="The number of frames to use in an FIR model.")
    config_arguments.add_argument("--fir_vars", nargs="*",
                        help="""A list of the task regressors to apply this FIR model to. The default is to apply it to all regressors if no
                        value is specified. A list must be specified if both types of models are being used""")
    config_arguments.add_argument("--hrf", nargs=2, type=int, metavar=("PEAK", "UNDER"),
                        help="""Two values to describe the hrf function that will be convolved with the task events. 
                        The first value is the time to the peak, and the second is the undershoot duration. Both in units of seconds.""")
    config_arguments.add_argument("--hrf_vars", nargs="*",
                        help="""A list of the task regressors to apply this HRF model to. The default is to apply it to all regressors if no
                        value is specifed. A list must be specified if both types of models are being used.""")
    config_arguments.add_argument("--unmodeled", "-um", nargs="*",
                        help="""A list of the task regressors to leave unmodeled, but still included in the final design matrix. These are
                        typically continuous variables that need not be modeled with hrf or fir, but any of the task regressors can be included.""")
    config_arguments.add_argument("--start_censoring", "-sc", type=int, default=0,
                        help="The number of frames to censor out at the beginning of each run. Typically used to censor scanner equilibrium time. Default is 0")
    config_arguments.add_argument("--confounds", "-c", nargs="+", default=[], 
                        help="A list of confounds to include from each confound timeseries tsv file.")
    config_arguments.add_argument("--fd_threshold", "-fd", type=float, default=0.9,
                        help="The framewise displacement threshold used when censoring high-motion frames")
    config_arguments.add_argument("--repetition_time", "-tr", type=float,
                        help="Repetition time of the function runs in seconds. If it is not supplied, an attempt will be made to read it from the JSON sidecar file.")
    config_arguments.add_argument("--detrend_data", "-dd", action="store_true", 
                        help="""Flag to demean and detrend the data before modeling. The default is to include
                        a mean and trend line into the nuisance matrix instead.""")
    config_arguments.add_argument("--no_global_mean", action="store_true",
                        help="Flag to indicate that you do not want to include a global mean into the model.")
    high_motion_params = config_arguments.add_mutually_exclusive_group()
    high_motion_params.add_argument("--spike_regression", "-sr", action="store_true",
                        help="Flag to indicate that framewise displacement spike regression should be included in the nuisance matrix.")
    high_motion_params.add_argument("--fd_censoring", "-fc", action="store_true",
                        help="Flag to indicate that frames above the framewise displacement threshold should be censored before the GLM.")
    config_arguments.add_argument("--run_exclusion_threshold", "-re", type=int, 
                        help="The percent of frames a run must retain after high motion censoring to be included in the fine GLM. Only has effect when '--fd_censoring' is active.")
    config_arguments.add_argument("--nuisance_regression", "-nr", action="store_true",
                        help="Flag to indicate that nuisance regression should be performed before performing the GLM for event-related activation.")
    config_arguments.add_argument("--nuisance_fd", "-nf", type=float,
                        help="The framewise displacement threshold used when censoring frames for nuisance regression.")
    config_arguments.add_argument("--highpass", "-hp", type=float, nargs="?", const=0.008,
                        help="""The high pass cutoff frequency for signal filtering. Frequencies below this value (Hz) will be filtered out. If the argument
                        is supplied but no value is given, then the value will default to 0.008 Hz""")
    config_arguments.add_argument("--lowpass", "-lp", type=float, nargs="?", const=0.1,
                        help="""The low pass cutoff frequency for signal filtering. Frequencies above this value (Hz) will be filtered out. If the argument
                        is supplied but no value is given, then the value will default to 0.1 Hz""")
    config_arguments.add_argument("--volterra_lag", "-vl", nargs="?", const=2, type=int,
                        help="""The amount of frames to lag for a volterra expansion. If no value is specified
                        the default of 2 will be used. Must be specifed with the '--volterra_columns' option.""")
    config_arguments.add_argument("--volterra_columns", "-vc", nargs="+", default=[],
                        help="The confound columns to include in the expansion. Must be specifed with the '--volterra_lag' option.")
    
    args = parser.parse_args()

    if args.hrf != None and args.fir != None:
        if not args.fir_vars or not args.hrf_vars:
            parser.error("Must specify variables to apply each model to if using both types of models")
    elif args.hrf == None and args.fir == None:
        parser.error("Must include model parameters for at least one of the models, fir or hrf.")
    
    if args.bold_file_type[0] != ".":
        args.bold_file_type = "." + args.bold_file_type
    if (args.bold_file_type in [".nii", ".nii.gz"]) and (not args.auto_mask or args.brain_mask or not args.brain_mask.is_file()):
        parser.error("If the bold file type is volumetric data, a valid '--brain_mask' option must also be supplied")
    elif args.bold_file_type == ".dtseries.nii" and args.brain_mask:
        args.brain_mask = None

    if (args.volterra_lag and not args.volterra_columns) or (not args.volterra_lag and args.volterra_columns):
        parser.error("The options '--volterra_lag' and '--volterra_columns' must be specifed together, or neither of them specified.")
    
    if callable(args.events_long):
        args.events_long = args.events_long(args)

    try:
        assert args.derivs_dir.is_dir(), "Derivatives directory must exist but is not found"
        assert args.raw_bids.is_dir(), "Raw data directory must exist but is not found"
    except AssertionError as e:
        logger.exception(e)
        exit_program_early(e)

    ################# Export the current arguments to a file #################
    if args.export_args:
        try:
            assert args.export_args.parent.exists() and args.export_args.suffix, "Argument export path must be a file path in a directory that exists"
            log_linebreak()
            logger.info(f"####### Exporting Configuration Arguments to: '{args.export_args}' #######\n")
            export_args_to_file(args, config_arguments, args.export_args)
        except Exception as e:
            logger.exception(e)
            exit_program_early(e)

    if not hasattr(args, "output_dir") or args.output_dir == None:
        args.output_dir = args.derivs_dir / f"{args.derivs_subfolder}/sub-{args.subject}/ses-{args.session}/func"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = args.output_dir.parent / "logs" 
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sub-{args.subject}_ses-{args.session}_task-{args.task}_desc-{datetime.datetime.now().strftime('%m-%d-%y_%I-%M%p')}.log"
    add_file_handler(logger, log_path)
    
    if args.debug:
        flags.debug = True
        logger.setLevel(logging.DEBUG)

    logger.info("Starting oceanfla...")
    logger.info(f"Log will be stored at {log_path}")

    ################# log the arguments used for this run #################
    for k,v in (dict(args._get_kwargs())).items():
        logger.info(f" {k} : {v}")
        
    model_type = "Mixed" if args.fir and args.hrf else "FIR" if args.fir else "HRF" 
    file_map_list = []

    try: 
        ################# find all preprocessed BOLD runs for this subject and session #################
        preproc_derivs = args.derivs_dir / args.preproc_subfolder
        bold_files = sorted(preproc_derivs.glob(f"**/*sub-{args.subject}_ses-{args.session}*task-{args.task}*bold{args.bold_file_type}"))
        assert len(bold_files) > 0, "Did not find any bold files in the given derivatives directory for the specified task and file type"
        
        ################# for each BOLD run, find the accompanying confounds file and events/events long file #################
        for bold_path in bold_files:
            file_map = {"bold" : bold_path}
            bold_base = bold_path.name.split("_space")[0]
            bold_base = bold_base.split("_desc")[0]

            confounds_search_path = f"{bold_base}_desc*-confounds_timeseries.tsv"
            confounds_files = list(bold_path.parent.glob(confounds_search_path))
            assert len(confounds_files) == 1, f"Found more or less than one confounds file for bold run: {str(bold_path)} search path: {confounds_search_path} len: {len(confounds_files)}"
            file_map["confounds"] = confounds_files[0]

            if args.events_long:
                events_long_search_path = f"{bold_base}*_desc*events_long.csv"
                events_long_files = list(args.events_long.glob(f"**/{events_long_search_path}"))
                assert len(events_long_files) == 1, f"Found more or less than one events long file for bold run: {str(bold_path)} search path: {events_long_search_path} len: {len(events_long_files)}"
                file_map["events"] = events_long_files[0]
            else:
                event_search_path = f"{bold_base}*_events.tsv"
                event_files = list(args.raw_bids.glob("**/" + event_search_path))
                assert len(event_files) == 1, f"Found more or less than one event file for bold run: {str(bold_path)} search path: {event_search_path} len: {len(event_files)}"
                file_map["events"] = event_files[0]

            file_map_list.append(file_map)

        tr = args.repetition_time
        img_header = None
        trial_types = set()
        func_data_list = []
        design_df_list = []
        noise_df_list = []
        mask_list = []

        # For each set of run files, create the design matrix and finalize the BOLD data before the session-task level GLM
        for i, run_map in enumerate(file_map_list):
            log_linebreak()
            logger.info(f"processing bold file: {run_map['bold']}")
            logger.info(f" loading in BOLD data")

            run_info = len(str(run_map['bold']).split('run-')) > 1
            run_num = i+1
            if run_info:
                run_num = int(run_map['bold'].name.split('run-')[-1].split('_')[0])
            run_info = f'run-{run_num:02d}_'

            func_data, read_tr, read_header = load_data(
                func_file=run_map['bold'], 
                brain_mask=args.brain_mask,
                need_tr=(not tr),
            )

            tr = tr if tr else read_tr
            img_header = img_header if img_header else read_header


            ################ create the events matrix #################
            logger.info(" reading events file and creating design matrix")
            events_long = None
            if args.events_long:
                events_long = pd.read_csv(run_map["events"], index_col=0)
            else:
                events_long = make_events_long( 
                    event_file=run_map["events"], 
                    volumes=func_data.shape[0],
                    tr=tr, 
                )
            events_df, run_conditions = events_to_design(   
                events_long=events_long,
                fir=args.fir,
                fir_list=args.fir_vars if args.fir_vars else None,
                hrf=args.hrf,
                hrf_list=args.hrf_vars if args.hrf_vars else None,
                unmodeled_list=args.unmodeled
            )


            ################# create the noise matrix #################
            logger.info(" reading confounds file and creating nuisance matrix")
            noise_df = make_noise_ts(
                confounds_file=run_map["confounds"],
                confounds_columns=args.confounds,
                demean=(not args.detrend_data), 
                linear_trend=(not args.detrend_data),
                spike_threshold=args.fd_threshold if args.spike_regression else None,
                volterra_expansion=args.volterra_lag,
                volterra_columns=args.volterra_columns
            )


            ################# create and apply the acqusition mask #################
            acquisition_mask = np.ones(shape=(func_data.shape[0],)).astype(bool)
            acquisition_mask[:args.start_censoring] = False
            if args.start_censoring > 0:
                logger.info(f" removing the first {args.start_censoring} frames from the beginning of the run")
            func_data = func_data[acquisition_mask, :]
            events_df = events_df.loc[acquisition_mask, :]
            noise_df = noise_df.loc[acquisition_mask, :]


            ################# create high motion mask and exclude run if needed #################
            run_mask = np.ones(shape=(func_data.shape[0],)).astype(bool)
            if args.fd_censoring:
                logger.info(f" censoring timepoints using a high motion mask with a framewise displacement threshold of {args.fd_threshold}")
                confounds_df = pd.read_csv(run_map["confounds"], sep="\t")
                fd_mask = confounds_df.loc[:, "framewise_displacement"].to_numpy()[acquisition_mask] < args.fd_threshold
                run_mask &= fd_mask
                logger.info(f" a total of {np.sum(~run_mask)} timepoints will be censored with this framewise displacement threshold")
                frame_retention_percent = (np.sum(run_mask) / run_mask.shape[0]) * 100
                logger.info(f" total run length after start censoring: {run_mask.shape[0]}, number of frames retained after high motion censoring: {np.sum(run_mask)}")
                # if censoring causes the number of retained frames to be below the run exclusion threshold, drop the run
                if args.run_exclusion_threshold and (frame_retention_percent < args.run_exclusion_threshold):
                    logger.info(f" BOLD run: {run_map['bold']} has fell below the run exclusion threshold of {args.run_exclusion_threshold}% and will not be used in the final GLM.")
                    continue


            ################# save out the nuisance matrix and events matrix (if debug)#################
            noise_df_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_{run_info}desc-model-{model_type}_nuisance.csv"
            logger.info(f" saving nuisance matrix to file: {noise_df_filename}")
            noise_df.to_csv(noise_df_filename)
            if flags.debug:
                events_long_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_{run_info}desc-events_long.csv"
                logger.debug(f" saving events long to file: {events_long_filename}")
                events_long.to_csv(events_long_filename)

                events_df_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_{run_info}desc-model-{model_type}-events_matrix.csv"
                logger.debug(f" saving events matrix to file: {events_df_filename}")
                events_df.to_csv(events_df_filename)


            ################# detrend the BOLD data if specifed #################
            if args.detrend_data:
                logger.info(" detrending the BOLD data")
                func_data_detrend = demean_detrend(
                    func_data=func_data
                )
                run_map["data_detrend"] = func_data_detrend
                func_data = func_data_detrend
                if flags.debug: 
                    cleanimg, img_suffix = create_image(
                        data=func_data,
                        brain_mask=args.brain_mask,
                        tr=tr,
                        header=img_header
                    )
                    cleaned_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_{run_info}desc-model-{model_type}_detrended{img_suffix}"
                    logger.debug(f" saving BOLD data after detrending to file: {cleaned_filename}")
                    nib.save(
                        cleanimg,
                        cleaned_filename
                    )
            

            ################# nuisance regression if specified #################
            if args.nuisance_regression:
                nuisance_mask = np.ones(shape=(func_data.shape[0],)).astype(bool)
                if args.nuisance_fd:
                    logger.info(f" censoring timepoints for nuisance regression using the framewise displacement threshold of {args.nuisance_fd}")
                    confounds_df = pd.read_csv(run_map["confounds"], sep="\t")
                    nuisance_fd_mask = confounds_df.loc[:, "framewise_displacement"].to_numpy()[acquisition_mask] < args.nuisance_fd
                    nuisance_mask &= nuisance_fd_mask
                    logger.info(f" a total of {np.sum(~nuisance_mask)} timepoints will be censored with this nuisance framewise displacement threshold")

                logger.info(" performing nuisance regression")
                nuisance_betas, func_data_residuals = massuni_linGLM(
                    func_data=func_data,
                    design_matrix=noise_df,
                    mask=nuisance_mask
                )

                run_map["data_resids"] = func_data_residuals
                func_data = func_data_residuals

                if flags.debug:
                    # save out the BOLD data after nuisance regression 
                    nrimg, img_suffix = create_image(
                        data=func_data,
                        brain_mask=args.brain_mask,
                        tr=tr,
                        header=img_header
                    )
                    nr_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_{run_info}desc-model-{model_type}_nuisance-regressed{img_suffix}"
                    logger.debug(f" saving BOLD data after nuisance regression to file: {nr_filename}")
                    nib.save(
                        nrimg,
                        nr_filename
                    )

                    # save out the nuisance betas
                    for i, noise_col in enumerate(noise_df.columns):
                        beta_img, img_suffix = create_image(
                            data=np.expand_dims(nuisance_betas[i,:], axis=0),
                            brain_mask=args.brain_mask,
                            tr=tr,
                            header=img_header
                        )
                        beta_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_{run_info}desc-model-{model_type}-beta-{noise_col}-frame-0{img_suffix}"
                        logger.debug(f" saving betas for nuisance variable: {noise_col} to file: {beta_filename}")
                        nib.save(
                            beta_img,
                            beta_filename
                        )


            ################# filter the data if specified #################
            if args.lowpass or args.highpass:    
                logger.info(f" filtering the BOLD data with a highpass of {args.highpass} and a lowpass of {args.lowpass}")
                func_data_filtered = filter_data(
                    func_data=func_data,
                    mask=run_mask,
                    tr=tr,
                    low_pass=args.lowpass if args.lowpass else None,
                    high_pass=args.highpass if args.highpass else None,

                )
                run_map["data_filtered"] = func_data_filtered
                func_data = func_data_filtered
                if flags.debug: 
                    cleanimg, img_suffix = create_image(
                        data=func_data,
                        brain_mask=args.brain_mask,
                        tr=tr,
                        header=img_header
                    )
                    filtered_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_{run_info}desc-model-{model_type}_filtered{img_suffix}"
                    logger.debug(f" saving BOLD data after filtering to file: {filtered_filename}")
                    nib.save(
                        cleanimg,
                        filtered_filename
                    )
            

            ################# apppend the run-wise data to the session list #################
            assert func_data.shape[0] == len(noise_df), "The functional data and the nuisance matrix have a different number of timepoints"
            if not args.nuisance_regression:
                logger.info(" appending the nuisance matrix to the run list")
                noise_df_list.append(noise_df)

            logger.info(" appending the high motion mask to the run list")
            assert func_data.shape[0] == run_mask.shape[0] , "The functional data and the high motion mask have a different number of timepoints"
            mask_list.append(run_mask)

            logger.info(" appending the BOLD data and design matrix to the run list")
            trial_types.update(run_conditions)
            assert func_data.shape[0] == len(events_df), "The functional data and the design matrix have a different number of timepoints"
            func_data_list.append(func_data)
            design_df_list.append((events_df,run_num))
            logger.info(f" a total of {np.sum(run_mask)} frames will be used from BOLD file: {run_map['bold']}")

        
        log_linebreak()
        ################# check that at least one run passed the run exculsion threshold #################
        assert len(func_data_list) == len(design_df_list) and (len(noise_df_list) == 0 or (len(noise_df_list) == len(func_data_list))), "Something went wrong! Run-wise data lists are not the same size."
        if len(func_data_list) == 0:
            logger.info(f"all run have been excluded with the run exclusion threshold of {args.run_exclusion_threshold}%, try lowering this parameter before running again!")
            return

        ################# create the final design matrix and append the data together #################
        logger.info("concatenating run level BOLD data and design matrices for GLM")
        final_func_data, final_design_unmasked = create_final_design(
            data_list=func_data_list,
            design_list=design_df_list,
            noise_list=noise_df_list if len(noise_df_list) == len(func_data_list) else None,
            exclude_global_mean=args.no_global_mean
        )
        final_high_motion_mask = np.concat(mask_list, axis=0)
        logger.info(f"total number of framess after start censoring for each run: {final_high_motion_mask.shape[0]}")
        logger.info(f"total number of frames that will be used in the final GLM after high motion censoring: {np.sum(final_high_motion_mask)}")

        final_design_masked = final_design_unmasked.loc[final_high_motion_mask, :]
        final_design_unmasked_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_desc-model-{model_type}-design_unmasked.csv"
        logger.info(f"saving the final unmasked design matrix to file: {final_design_unmasked_filename}")
        final_design_unmasked.to_csv(final_design_unmasked_filename)
        final_design_masked_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_desc-model-{model_type}-design_final.csv"
        logger.info(f"saving the final design matrix to file: {final_design_masked_filename}")
        final_design_masked.to_csv(final_design_masked_filename)


        ################# run the final GLM #################
        logger.info("running GLM on concatenated BOLD data with final design matrix")
        
        activation_betas, func_residual = massuni_linGLM(
            func_data=final_func_data,
            design_matrix=final_design_unmasked,
            mask=final_high_motion_mask
        )

        ################# save out the beta values #################
        logger.info("saving betas from GLM into files")
        fir_betas_to_combine = set()
        for i, c in enumerate(final_design_unmasked.columns):
            if args.fir and c[-3] == "_" and c[-2:].isnumeric() and c[:-3] in trial_types:
                fir_betas_to_combine.add(c[:-3])
                continue
            elif c in trial_types:
                beta_img, img_suffix = create_image(
                    data=np.expand_dims(activation_betas[i,:], axis=0),
                    brain_mask=args.brain_mask,
                    tr=tr,
                    header=img_header
                )
                beta_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_desc-model-{model_type}-beta-{c}-frame-0{img_suffix}"
                logger.info(f" saving betas for variable {c} to file: {beta_filename}")
                nib.save(
                    beta_img,
                    beta_filename
                )

        if args.fir:
            for condition in fir_betas_to_combine:
                beta_frames = np.zeros(shape=(args.fir, activation_betas.shape[1]))
                for f in range(args.fir):
                    beta_column = final_design_unmasked.columns.get_loc(f"{condition}_{f:02d}")
                    beta_frames[f,:] = activation_betas[beta_column,:]
                    beta_img, img_suffix = create_image(
                        data=np.expand_dims(activation_betas[beta_column,:], axis=0),
                        brain_mask=args.brain_mask,
                        tr=tr,
                        header=img_header
                    )
                    beta_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_desc-model-{model_type}-beta-{condition}-frame-{f}{img_suffix}"
                    logger.info(f" saving betas for variable {condition} frame {f} to file: {beta_filename}")
                    nib.save(
                        beta_img,
                        beta_filename
                    )

                beta_img, img_suffix = create_image(
                    data=beta_frames,
                    brain_mask=args.brain_mask,
                    tr=tr,
                    header=img_header
                )
                beta_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_desc-model-{model_type}-beta-{condition}-concatenated{img_suffix}"
                logger.info(f" saving betas for variable {condition} (all {args.fir} modeled frames) to file: {beta_filename}")
                nib.save(
                    beta_img,
                    beta_filename
                )

        ################# save out the nuisance betas and the residuals #################
        if flags.debug:
            # save out the nuisance betas
            nuisance_cols = [(i,c) for i,c in enumerate(final_design_unmasked.columns) if len([condition for condition in trial_types if condition in c])==0]
            for i, noise_col in nuisance_cols:
                beta_img, img_suffix = create_image(
                    data=np.expand_dims(activation_betas[i,:], axis=0),
                    brain_mask=args.brain_mask,
                    tr=tr,
                    header=img_header
                )
                beta_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_desc-model-{model_type}-beta-{noise_col}-frame-0{img_suffix}"
                logger.debug(f" saving betas for nuisance variable: {noise_col} to file: {beta_filename}")
                nib.save(
                    beta_img,
                    beta_filename
                )

            # save out residuals of GLM
            resid_img, img_suffix = create_image(
                data=func_residual,
                brain_mask=args.brain_mask,
                tr=tr,
                header=img_header
            )
            resid_filename = args.output_dir/f"sub-{args.subject}_ses-{args.session}_task-{args.task}_desc-model-{model_type}_residual{img_suffix}"
            logger.debug(f" saving residual BOLD data after final GLM to file: {resid_filename}")
            nib.save(
                resid_img,
                resid_filename
            )

    except Exception as e:
        logger.exception(e, stack_info=True)
        exit_program_early(str(e))
    
    logger.info("oceanfla complete!")

if __name__ == "__main__":
    main()

