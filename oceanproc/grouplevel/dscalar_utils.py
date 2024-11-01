import re
import os
from glob import glob
from pathlib import Path
import nibabel as nib
import numpy as np


def _find_all_dscalars(func_dir: str|Path) -> list[str]:
    """
    Return a list of all paths to dscalars in func_dir.
    :param func_dir: path to a subject's functional derivatives directory
    :type func_dir: pathlib.Path or str
    :return: a list of absolute paths to dscalars
    :rtype: list[str]
    """
    rel_paths = glob(os.path.join(func_dir, "*dscalar.nii"))
    return [os.path.abspath(p) for p in rel_paths]

def _filter_dscalar_paths_by_regressor(dscalar_paths: list[str],
                                      regressors: list[str]) -> list[str]:
    """
    Returns a list of paths to .dscalar.nii files containing the names of given regressors.
    :param dscalar_paths: list of absolute paths to a set of dscalar files
    :type dscalar_paths: list[str]
    :param regressors: list of names of regressors contained in the dscalar base name
    :type regressors: list[str]
    """
    if len(regressors) < 1:
        raise ValueError("List of regressor names must not be empty.")
    filtered_dscalars = []
    for regressor in regressors:
        filtered_dscalars.extend([p for p in dscalar_paths if re.search(regressor, os.path.basename(p))])
    return filtered_dscalars

def _load_dscalars(dscalar_paths: list[str]) -> list[nib.cifti2.cifti2.Cifti2Image]:
    assert len(dscalar_paths) > 0, "Lenght of dscalar paths list must be greater than 0"
    dscalars = []
    for p in dscalar_paths:
        dscalars.append(nib.load(p))
    return dscalars

def _combine_dscalars(dscalars: list[nib.cifti2.cifti2.Cifti2Image]) -> np.typing.ArrayLike:
    assert len(dscalars) > 0, "Length of dscalars list must be greater than 0"
    header = dscalars[0].header
    combined_data = np.array([img.get_fdata().squeeze() for img in dscalars])
    scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis([f"betamap{i+1}" for i in range(combined_data.shape[0])])
    brain_model_axis = header.get_axis(1)
    combined_img = nib.Cifti2Image(combined_data, (scalar_axis, brain_model_axis))
    return combined_img

