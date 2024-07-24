#!/usr/bin/env python3
import numpy as np
import os
from glob import glob
from pathlib import Path
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import nibabel as nib
# from nilearn.glm.first_level import FirstLevelModel
# from nilearn.plotting import plot_design_matrix
# import matplotlib.pyplot as plt
import nilearn.masking as nmask
from nilearn.signal import clean
import json
from scipy import signal
from ..oceanparse import OceanParser


"""
TODO: 
    * Find good way to pass hrf peak and undershoot variables
    * Consult on solid way to implement band pass filtering
    * Find way to implement Volterra expansion for noise dataframe

"""

def make_option(value, key=None, delimeter=" "):
    """
    Generate a string, representing an option that gets fed into a subprocess.

    For example, if a key is 'option' and its value is True, the option string it will generate would be:

        --option

    If value is equal to some string 'value', then the string would be:

        --option value

    If value is a list of strings:

        --option value1 value2 ... valuen

    :param key: Name of option to pass into a subprocess, without double hyphen at the beginning.
    :type key: str
    :param value: Value to pass in along with the 'key' param.
    :type value: str or bool or list[str] or None
    :param delimeter: character to separate the key and the value in the option string. Default is a space.
    :type delimeter: str
    :return: String to pass as an option into a subprocess call.
    :rtype: str
    """
    second_part = None
    if type(value) == bool and value:
        second_part = " "
    if type(value) == list:
        second_part = f"{delimeter}{delimeter.join(value)}"
    elif type(value) == str or type(value) == int or type(value) == float:
        second_part = f"{delimeter}{value}"
    else:
        return ""
    return f"--{key.replace('_', '-')}{second_part}" if key else second_part 


def load_data(func_file: str, brain_mask: str = None) -> np.ndarray:
    tr = None
    sidecar_file = func_file.split(".")[0] + ".json"
    assert os.path.isfile(sidecar_file), f"Cannot find the .json sidecar file for bold run: {func_file}"
    with open(sidecar_file, "r") as f:
        jd = json.load(f)
        tr = jd["RepetitionTime"]

    if func_file.endswith(".dtseries.nii"):
        return (nib.load(func_file).get_fdata(), tr)
    elif func_file.endswith(".nii") or func_file.endswith(".nii.gz"):
        if brain_mask:
            return (nmask.apply_mask(func_file, brain_mask), tr)
        else:
            # raise Exception("Volumetric data must also have an accompanying brain mask")
            return None
        

def create_image(data: npt.ArrayLike, brain_mask: str = None, tr: float = None):
    img = None
    suffix = ".nii"
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
        ax1 = nib.cifti2.cifti2_axes.BrainModelAxis(
            name=(['CIFTI_STRUCTURE_CORTEX_LEFT']*(data.shape[1]/2))+(['CIFTI_STRUCTURE_CORTEX_RIGHT']*(data.shape[1]/2)),
            vertex=np.arange(data.shape[1]/2),
            nvertices={'CIFTI_STRUCTURE_CORTEX_LEFT':data.shape[1]/2, 'CIFTI_STRUCTURE_CORTEX_RIGHT':data.shape[1]/2}
        )
        img = nib.cifti2.cifti2.Cifti2Image(data, (ax0, ax1))
    return (img ,suffix)


def demean_detrend(func_data: npt.ArrayLike) -> np.ndarray:
    data_dd = signal.detrend(func_data, axis=0, type = 'linear')
    return data_dd


def hrf(time, time_to_peak=5, undershoot_dur=12):
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

    from scipy.stats import gamma

    peak = gamma.pdf(time, time_to_peak)
    undershoot = gamma.pdf(time, undershoot_dur)
    hrf_timeseries = peak - 0.35 * undershoot
    return hrf_timeseries


def hrf_convolve_features(features, column_names='all', time_col='index', units='s', time_to_peak=5, undershoot_dur=12):
    """
    This function convolves a hemodynamic response function with each column in a timeseries dataframe.

    Parameters
    ----------
    features: DataFrame
        A Pandas dataframe with the feature signals to convolve.
    column_names: list
        List of columns names to use.  Default is "all"
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
    if column_names == 'all':
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
    hrf_sig = hrf(time, time_to_peak=time_to_peak, undershoot_dur=undershoot_dur)
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


def make_noise_ts(confounds_file: str, 
                  confound_columns: list, 
                  demean: bool = False, 
                  linear_trend: bool = False, 
                  spike_threshold: float = None,
                  volterra_expansion: int = None,
                  volterra_columns: list = None
                  ):
    fd = "framewise_displacement"
    select_columns = set(confound_columns)
    select_columns.update(volterra_columns)
    nuisance = pd.read_csv(confounds_file, delimiter='\t').loc[:,select_columns]
    if fd in select_columns:
        nuisance.loc[0, fd] = 0

    if demean: 
        nuisance["mean"] = 1

    if linear_trend:
        nuisance["trend"] = np.arange(0, len(nuisance))

    if spike_threshold:
        b = 0
        for a in range(len(nuisance)):
            if nuisance.loc[a,fd] > spike_threshold:
                nuisance[f"spike{b}"] = 0
                nuisance.loc[a, f"spike{b}"] = 1
                b += 1

    if volterra_expansion and volterra_columns:
        for vc in volterra_columns:
            for lag in range(volterra_expansion):
                nuisance.loc[:, f"{vc}_{lag+1}"] = nuisance.loc[:, vc].shift(lag+1)
        nuisance.fillna(0, inplace=True)

    return nuisance


def events_to_design(func_data: npt.ArrayLike, tr: float, event_file: str, fir: int = None, assumed: list[int] = None, design_file: str = None):
    duration = tr * func_data.shape[0]
    events_df = pd.read_csv(event_file, index_col=None, delimiter='\t')
    conditions = [s for s in np.unique(events_df.trial_type)]
    events_long = pd.DataFrame(0, columns=conditions, index=np.arange(0, duration, tr))

    for e in events_df.index:
        i = find_nearest(events_long.index, events_df.loc[e,'onset'])
        events_long.loc[i, events_df.loc[e,'trial_type']] = 1
        if events_df.loc[e,'duration'] > tr:
            offset = events_df.loc[e,'onset'] + events_df.loc[e,'duration']
            j = find_nearest(events_long.index, offset)
            if i>j:
                events_long.loc[j, events_df.loc[e,'trial_type']] = 1
                # need to add to fill in time in between

    if fir:
        col_names = {c:c+"_00" for c in conditions}
        events_long = events_long.rename(columns=col_names)
        for c in conditions:
            for i in range(1, fir):
                events_long.loc[:,f"{c}_{i:02d}"] = np.array(np.roll(events_long.loc[:,col_names[c]], shift=i, axis=0))
                # so events do not roll back around to the beginning
                events_long.loc[:i,f"{c}_{i:02d}"] = 0
        events_long = events_long.astype(int)
    elif assumed:
        cfeats = hrf_convolve_features(features=events_long, 
                                       column_names=conditions,
                                       time_to_peak=assumed[0],
                                       undershoot_dur=assumed[1])
        for c in conditions:
            events_long[c] = cfeats[c]
        pass
    
    if design_file:
        events_long.to_csv(design_file)
    
    return (events_long, conditions)



def bandpass_filter(func_data: npt.ArrayLike, 
                    tr: float, 
                    high_cut: float = 0.1, 
                    low_cut: float = 0.008,
                    order: int = 2 ):
    fs = 1/tr
    nyquist = 1/(2*tr)
    high = high_cut/nyquist
    low = low_cut/nyquist
    # sos = signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    b, a = signal.butter(order, [low, high], btype="band", fs=fs)

    # filtered_data = signal.sosfiltfilt(sos=sos, x=func_data, axis=0)
    filtered_data = signal.filtfilt(b=b, a=a, x=func_data, axis=0)
    
    return filtered_data

"""
SPECTRAL INTERPOLATION FOR MOTION SPIKE FRAMES
"""



"""
REGRESS OUT NUISANCE VARIABLES
"""
def nuisance_regression(func_data: npt.ArrayLike, noise_matrix: pd.DataFrame, fd_thresh: float = None):
    ss = StandardScaler()
    # designmat = ss.fit_transform(noise_matrix[noise_matrix["framewise_displacement"]<fd_thresh].to_numpy())
    designmat = ss.fit_transform(noise_matrix.to_numpy())
    neuro_data = ss.fit_transform(func_data)
    inv_mat = np.linalg.pinv(designmat)
    beta_data = np.dot(inv_mat, neuro_data)
    est_values = np.dot(designmat, beta_data)

    return func_data - est_values


"""
COMBINE DESIGN MATRICES 
"""
def create_final_design(data_list: list[npt.ArrayLike], design_list: list[pd.DataFrame], noise_list: list[pd.DataFrame] = None):
    num_runs = len(data_list)
    assert num_runs == len(design_list), "There should be the same number of design matrices and functional runs"
    
    if noise_list:
        assert num_runs == len(noise_list), "There should be the same number of noise matrices and functional runs"
        for i in range(num_runs):
            noise_df = noise_list[i]
            rename_dict = dict()
            for c in noise_df.columns:
                if ("trend" in c) or ("mean" in c) or ("spike" in c):
                    rename_dict[c] = f"run-{i+1}_{c}"
            noise_df = noise_df.rename(columns=rename_dict)
            noise_list[i] = noise_df
            design_list[i] = pd.concat([design_list[i], noise_df], axis=1)

    final_design = pd.concat(design_list, axis=0, ignore_index=True)
    final_data = np.vstack(data_list)
    return (final_data, final_design)



# MODIFY FUNCTION
def massuni_linGLM(func_data: npt.ArrayLike, design_matrix: pd.DataFrame):
    ss = StandardScaler()
    design_matrix = ss.fit_transform(design_matrix.to_numpy())
    neuro_data = ss.fit_transform(func_data)

    inv_mat = np.linalg.pinv(design_matrix)
    beta_data = np.dot(inv_mat, neuro_data)
    return beta_data



def main():

    parser = OceanParser(
        prog="oceanfla",
        description="Ocean Labs first level analysis",
        fromfile_prefix_chars="@",
        epilog="An arguments file can be accepted with @FILEPATH"
    )
    
    parser.add_argument("--subject", "-su",
                        help="The subject ID")
    parser.add_argument("--session", "-se",
                        help="The session ID")
    parser.add_argument("--task", "-t", #required=True,
                        help="The name of the task to analyze.")
    parser.add_argument("--bold_file_type", "-ft", #required=True,
                        help="The file type of the functional runs to use.")
    parser.add_argument("--brain_mask", "-bm", type=Path,
                        help="If the bold file type is volumetric data, a brain mask must also be supplied.")
    parser.add_argument("--derivs_dir", "-d", type=Path, #required=True,
                        help="Path to the BIDS formatted derivatives directory for this subject and session.")
    parser.add_argument("--raw_bids", "-r", type=Path, #required=True,
                        help="Path to the BIDS formatted raw data directory for this subject and session.")
    parser.add_argument("--output_dir", "-o", type=Path, #required=True,
                        help="Path to the directory to store the results of this analysis")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--fir_frames", "-ff", type=int,
                             help="The number of frames to use in an FIR model.")
    model_group.add_argument("--hrf", nargs=2, type=int, metavar=("PEAK", "UNDER"),
                             help="""Two values to describe the hrf function that will be convolved with the task events. 
                             The first value is the time to the peak, and the second is the undershoot duration. Both in units of seconds.""")
    parser.add_argument("--confounds", "-c", nargs="+", #required=True,
                        help="A list of confounds to include from each confound timeseries tsv file.")
    parser.add_argument("--fd_threshold", "-fd", type=float, 
                        help="The framewise displacement threshold used when censoring high-motion frames")
    parser.add_argument("--detrend_data", "-dd", action="store_true", 
                        help="""Flag to demean and detrend the data before modeling. The default is to include
                        a mean and trend line into the nuisance matrix instead.""")
    parser.add_argument("--spike_censoring", "-sc", action="store_true",
                        help="Flag to indicate that framewise displacement spike censoring should be included in the nuisance matrix.")
    parser.add_argument("--nuisance_regression", "-nr", action="store_true",
                        help="Flag to indicate that nuisance regression should be performed before performing the GLM for event-related activation.")
    parser.add_argument("--bp_filter", "-bf", type=float, nargs="*",
                        help="""Frequency parameters for a bandpass filter. First value is the high cut-off and 
                        the second value is the low cut-off. Both is units of HZ. If no values are specified, the defaults of 
                        0.1 and 0.008 will be used.""")
    parser.add_argument("--volterra_lag", "-vl", nargs="?", const=2, type=int,
                        help="""The amount of frames to lag for a volterra expansion. If no value is specified
                        the default of 2 will be used. Must be specifed with the '--volterra_columns' option.""")
    parser.add_argument("--volterra_columns", "-vc", nargs="+",
                        help="The confound columns to include in the expansion. Must be specifed with the '--volterra_lag' option.")
    parser.add_argument("--export_args", "-ea", 
                        help="Path to a file to save the current arguments.")
    
    args = parser.parse_args()

    if args.bp_filter != None:
        if len(args.bp_filter) == 0:
            args.bp_filter = [0.1, 0.008]
        elif len(args.bp_filter) != 0 and len(args.bp_filter) != 2:
            parser.error("Expecting 0 or 2 arguments for the '--bp_filter' option")
    
    if args.bold_file_type[0] != ".":
        args.bold_file_type = "." + args.bold_file_type
    if (args.bold_file_type == ".nii" or args.bold_file_type == ".nii.gz") and (not args.brain_mask or not args.brain_mask.is_file()):
        parser.error("If the bold file type is volumetric data, a valid '--brain_mask' option must also be supplied")

    if (args.volterra_lag != None and args.volterra_columns == None) or (args.volterra_lag == None and args.volterra_columns != None):
        parser.error("The options '--volterra_lag' and '--volterra_columns' must be specifed together, or neither of them specified.")


    breakpoint()
    ##### Export the current arguments to a file #####
    if args.export_args:
        print(f"####### Exporting Arguments to: '{args.export_args}' #######")
        all_opts = dict(args._get_kwargs())
        opts_to_save = dict()
        for group in parser._action_groups:
            for a in group._group_actions:
                if a.dest in all_opts and all_opts[a.dest]:
                    if type(all_opts[a.dest]) == bool:
                        opts_to_save[a.option_strings[0]] = ""
                        continue
                    elif isinstance(all_opts[a.dest], Path):
                        opts_to_save[a.option_strings[0]] = all_opts[a.dest].as_posix()
                        continue
                    opts_to_save[a.option_strings[0]] = all_opts[a.dest]
        with open(args.export_args, "w") as f:
            if args.export_args.endswith(".json"):
                f.write(json.dumps(opts_to_save))
            else:
                for k,v in opts_to_save.items():
                    f.write(f"{k}{make_option(value=v)}\n")


    # Find all of the functional runs for given task
    # Find all of the confounds files for the given task
    # Find all of the event files of the given task

    # For each functional run
    ## Load in the functional data
    ## Create the design matrix
    ## Create the noise matrix
    ## Nuisance regression -- optional
    ## Bandpass filtering -- optional

    # Combine design matrices (and noise matrices if no nuisance regression was performed)
    # Run GLM

    assert args.derivs_dir.is_dir(), "Derivatives directory must exist"
    assert args.raw_bids.is_dir(), "Raw data directory must exist"

    bold_files = sorted(args.derivs_dir.glob(f"**/*sub-{args.subject}_ses-{args.session}*task-{args.task}*bold*{args.bold_file_type}"))
    assert len(bold_files) > 0, "Did not find any bold files in the given derivatives directory for the specified task and file type"

    file_map_list = []

    for bold_path in bold_files:
        bold_base = bold_path.name.split("_space")[0]
        bold_base = bold_base.split("_desc")[0]
        # confound_name = bold_base + "_desc-confounds_timeseries.tsv"
        confound_path = bold_path.parent / f"{bold_base}_desc-confounds_timeseries.tsv"
        assert confound_path.is_file(), f"Cannot find a confounds file for bold run: {str(bold_path)}"
        event_search_path = f"{bold_base}*_events.tsv"
        event_files = args.raw_bids.glob(event_search_path)
        assert len(event_files) == 1, f"Found more or less than one event file for bold run: {str(bold_path)}"
        events_path = event_files[0]
        file_map_list.append({
            "bold": bold_path,
            "confounds": confound_path,
            "events": events_path
        })

    tr = None
    trial_types = set()
    func_data_list = []
    design_df_list = []
    noise_df_list = []
    breakpoint()
    for run_map in file_map_list:
        func_data, tr = load_data(run_map["bold"].as_posix(), args.brain_mask)

        events_df, run_conditions = events_to_design(   
            func_data=func_data,
            tr=tr,
            event_file=run_map["events"],
            fir=args.fir_frams if args.fir_frames else None,
            assumed=args.hrf if args.hrf else None,
        )

        trial_types.update(run_conditions)

        noise_df = make_noise_ts(
            confounds_file=run_map["confounds"],
            confound_columns=args.confound_columns,
            demean=(not args.detrend_data), 
            linear_trend=(not args.detrend_data),
            spike_threshold=args.fd_threshold if args.spike_censoring else None,
            volterra_expansion=args.volterra_lag,
            volterra_columns=args.volterra_columns
        )

        if args.nuisance_regression:
            func_data_residuals = nuisance_regression(
                func_data=func_data,
                noise_matrix=noise_df,
                fd_thresh=args.fd_thresh
            )
            run_map["data_resids"] = func_data_residuals
            func_data = func_data_residuals
        else:
            noise_df_list.append(noise_df)

        if args.bp_filter:
            sample_mask = noise_df.loc[:, "framewise_displacement"].to_numpy()
            sample_mask = sample_mask > args.fd_thresh
            func_data_filtered = clean(
                signals=func_data,
                detrend=args.detrend_data,
                sample_mask=sample_mask,
                # confounds=noise_df,
                filter="butterworth",
                low_pass=args.bp_filter[0],
                high_pass=args.bp_filter[1],
                t_r=tr,
            )
            run_map["data_filtered"] = func_data_filtered
            func_data = func_data_filtered
        elif args.detrend_data:
            func_data_detrend = demean_detrend(
                func_data=func_data
            )
            run_map["data_detrend"] = func_data_detrend
            func_data = func_data_detrend

        func_data_list.append(func_data)
        design_df_list.append(events_df)
        

    final_func_data, final_design_df = create_final_design(
        data_list=func_data_list,
        design_list=design_df_list,
        noise_list=noise_df_list if len(noise_df_list) == len(func_data_list) else None
    )
    
    breakpoint()

    activation_betas = massuni_linGLM(
        func_data=final_func_data,
        design_matrix=final_design_df
    )

    model_type = "FIR" if args.fir_frames else "HRF"

    for i, c in enumerate(final_design_df.columns):

        if args.fir_frames and c[:-3] in trial_types:
            continue
            # if c[:-3] in fir_concat:
            #     trial = c[:-3]
            #     fir_concat[trial].append()

        beta_img, img_suffix = create_image(
            data=np.expand_dims(activation_betas[i,:], axis=0),
            brain_mask=args.brain_mask,
            tr=tr
        )
       
        nib.save(
            beta_img,
            f"sub-{args.subject}_ses-{args.session}_task-{args.task}_desc-{model_type}activation-{c}{img_suffix}"
        )

    if args.fir_frames:
        for condition in trial_types:
            beta_frames = np.zeros(shape=(args.fir_frames, activation_betas.shape[1]))
            for f in range(args.fir_frames):
                beta_frames[f,:] = activation_betas[f"{condition}_{f:02d}",:]
            beta_img, img_suffix = create_image(
                data=beta_frames,
                brain_mask=args.brain_mask,
                tr=tr
            )
            nib.save(
                beta_img,
                f"sub-{args.subject}_ses-{args.session}_task-{args.task}_desc-{model_type}activation-{c}{img_suffix}"
            )
    

if __name__ == "__main__":
    main()
    

