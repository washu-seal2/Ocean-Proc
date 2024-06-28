import pandas as pd
import numpy as np
import nibabel as nib
import json
import os
from .utils import exit_program_early
from glob import glob
from pathlib import Path


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


def make_events_long(bold_run:str, event_file:str, output_file:str, tr:float):
    """
    Takes and event file and a funtional run and creates a long formatted events file
    that maps the onset of task events to a frame of the functional run

    :param bold_run: path to the function run corresponding to the event file
    :type bold_run: str
    :param event_file: path to the event timing file
    :type event_file: str
    :param output_file: file path (including name) to save the long formatted event file to
    :type output_file: str
    :param tr: Repetition time of the function run in seconds
    :type tr: float
    """
    nvols = nib.load(bold_run).dataobj.shape[-1]
    duration = nvols * tr

    events_df = pd.read_csv(event_file, index_col=None, delimiter="\t")
    conditions = [s for s in np.unique(events_df.trial_type)]
    events_long = pd.DataFrame(0, columns=conditions, index=np.arange(0,duration,tr))

    for e in events_df.index:
        i = find_nearest(events_long.index, events_df.loc[e, "onset"])
        events_long.loc[i, events_df.loc[e, "trial_type"]] = 1
        if events_df.loc[e, "duration"] > tr:
            offset = events_df.loc[e, "onset"] + events_df.loc[e, "duration"]
            j = find_nearest(events_long.index, offset)
            if j>i:
                events_long.loc[j, events_df.loc[e, "trial_type"]] = 1

    events_long.to_csv(output_file)


def append_to_confounds(confounds_file:str, fd_thresh:float):
    """
    Makes motion outlier regressors based on the framewise
    displacement threshold, and appends them to the confounds file
    for this functional run

    :param confounds_file: path to the confounds file for this functional run
    :type confounds_file: str
    :param fd_thresh: Framewise displacement threshold for this functional run
    :type fd_thresh: float
    """
    conf_df = pd.read_csv(confounds_file, delimiter="\t")
    b = 0
    for a in range(len(conf_df)):
        if conf_df.loc[a, "framewise_displacement"] > fd_thresh:
            conf_df[f"spike{b}"] = 0
            conf_df.loc[a, f"spike{b}"] = 1
            b += 1
    
    conf_df.to_csv(confounds_file, sep="\t")
    

def create_events_and_confounds(bids_path:str, derivs_path:str, sub:str, ses:str, fd_thresh:float):
    """
    Facilitates the creation of a long formatted events file
    and the appending of motion outlier regressors to the 
    confounds file for each functional run for this subject and session

    :param bids_path: Path to the bids directory containing the raw
    :type bids_path: str
    :param derivs_path: Path to BIDS-compliant derivatives folder.
    :type derivs_path: str
    :param subject: Subject id (
    :type subject: str
    :param session: Session id 
    :type session: str
    :param fd_thresh: Framewise displacement threshold for this functional run
    :type fd_thresh: float
    
    """

    print("####### Creating long formatted event files ########")

    bids_func = f"{Path(bids_path).as_posix()}/sub-{sub}/ses-{ses}/func"
    derivs_func = f"{Path(derivs_path).as_posix()}/sub-{sub}/ses-{ses}/func"
    if not os.path.isdir(bids_func):
        exit_program_early(f"Cannnot find 'func' - {bids_func} - bids directory for this subject and session")
    if not os.path.isdir(derivs_func):
        exit_program_early(f"Cannnot find 'func' - {derivs_func} - derivatives directory for this subject and session")

    event_time_files = glob(bids_func + "/*_events.tsv")
    print(f"Found {len(event_time_files)} event timing files")
    for etf in event_time_files:
        search_path = f"/sub-{sub}_ses-{ses}*"
        task = etf.split('task-')[-1].split('_')[0]
        search_path = f"{search_path}task-{task}*"
        run = None
        if "run" in etf:
            run = etf.split('run-')[-1].split('_')[0].zfill(2)
            search_path = f"{search_path}run-{run}*"
        bold_search_path = f"{bids_func}{search_path}bold.nii*"
        bold_file = glob(bold_search_path)
        if len(bold_file) < 1:
            print(f"Could not find any bold files that matched this event timing file: {etf}")
            continue
        confounds_search_path = f"{derivs_func}{search_path}confounds_timeseries.tsv"
        confounds_file = glob(confounds_search_path)
        if len(confounds_file) < 1:
            print(f"Could not find any confounds files that matched this event timing file: {etf}")
            continue
        confounds_file = confounds_file[0]
        bold_file = bold_file[0]
        side_car = bold_file.split(".")[0] + ".json"
        tr = None
        with open(side_car, "r") as f:
            jd = json.load(f)
            tr = jd["RepetitionTime"]

        event_file_out = f"{derivs_func}/sub-{sub}_ses-{ses}_task-{task}_{f'run-{run}' if run else ''}_desc-events_long.csv"
        make_events_long(bold_file, etf, event_file_out, tr)
        append_to_confounds(confounds_file, fd_thresh)
        




