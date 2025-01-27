import pandas as pd
import numpy as np
import re
from pathlib import Path

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


def design_helper(bids_path: Path, output_file: Path):
    assert bids_path.exists() and bids_path.is_dir(), "The supplied BIDS Path must be an existing directory"
    assert output_file.suffix == ".csv", "The supplied output file must be of type '.csv'"
    entry_list = []
    subs = sorted([d for d in bids_path.glob("sub-*") if d.is_dir()])
    for sub_dir in subs:
        ses_dirs = sorted([d for d in sub_dir.glob("ses-*") if d.is_dir()])
        for ses_dir in ses_dirs:
            beta_files = sorted([f for f in ses_dir.glob("func/sub-*_ses-*_task-*_desc-*model-*-beta-*-frame-*.nii*") if f.is_file()])
            for file in beta_files:
                name = file.name
                no_suffix = name.split(".")[0]
                underscore_split = no_suffix.split("_")
                sub = underscore_split[0].split("-")[-1]
                ses = underscore_split[1].split("-")[-1]
                task = underscore_split[2].split("task-")[-1]
                desc_split = [tk for part in underscore_split[3:] for tk in part.split("-")]
                model = "-".join(desc_split[desc_split.index("model")+1 : desc_split.index("beta")])
                beta = "-".join(desc_split[desc_split.index("beta")+1 : desc_split.index("frame")])
                frame = int(desc_split[desc_split.index("frame")+1])
                file_type = "surface" if name.split(".")[1] == "dscalar" else "volume"
                entry_list.append(
                    {
                        "sub" : sub,
                        "ses" : ses,
                        "task" : task,
                        "model" : model,
                        "beta" : beta,
                        "frame" : frame,
                        "file_type" : file_type,
                        "path" : str(file.resolve())
                    }
                )
    design_scaffold = pd.DataFrame(entry_list)
    design_scaffold.sort_values(by=["sub","ses","task","model","file_type","beta","frame"], inplace=True)
    design_scaffold.to_csv(output_file, index=False)


