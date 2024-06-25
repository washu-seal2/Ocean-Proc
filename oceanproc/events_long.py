import pandas as pd
import numpy as np
import nibabel as nib
import json
import os
from .utils import exit_program_early
from glob import glob
from pathlib import Path



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return(array[idx])


def make_events_long(bold_run:str, event_file:str, output_dir:str, tr:float):
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

    events_name = f"{output_dir}/{event_file.split('/')[-1].replace('events.tsv', 'events_long.csv')}"
    events_long.to_csv(events_name)


def append_to_confounds(confounds_file:str, fd_thresh:float):
    conf_df = pd.read_csv(confounds_file, delimiter="\t")
    b = 0
    for a in range(len(conf_df)):
        if conf_df.loc[a, "framewise_displacement"] > fd_thresh:
            conf_df[f"spike{b}"] = 0
            conf_df.loc[a, f"spike{b}"] = 1
            b += 1
    
    conf_df.to_csv(confounds_file, sep="\t")
    

def create_events_and_confounds(bids_path:str, derivs_path:str, sub:str, ses:str, fd_thresh:float):
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
        bold_file = glob(etf.replace("_events.tsv", "*_bold.nii*"))
        if len(bold_file) < 1:
            print(f"Could not find any bold files that matched this event timing file: {etf}")
            continue
        confounds_file = glob(f"{derivs_func}/{etf.replace('events','desc-confounds_timeseries')}")
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
        make_events_long(bold_file, etf, derivs_func, tr)
        append_to_confounds(confounds_file, fd_thresh)
        




