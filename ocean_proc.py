import argparse
import os
from pathlib import Path
from bids_wrapper import dicom_to_bids, exit_program_early
from group_series import map_fmap_to_func
import shlex
import shutil
from subprocess import Popen, PIPE


def make_option(key, value):
    first_part = f"--{key} "
    if value == None:
        return ""
    elif type(value) == bool and value:
        return first_part
    elif type(value) == list:
        return first_part + " ".join(value)
    elif type(value) == str:
        return first_part + value


def run_fmri_prep(subject:str, bids_path:Path, derivs_path:Path, option_chain:str):
    print("####### Starting fMRIPrep #######")
    if not bids_path.exists():
        exit_program_early(f"Bids path {bids_path} does not exist.")
    elif not derivs_path.exists():
        exit_program_early(f"Derivatives path {derivs_path} does not exist.")
    elif shutil.which('fmriprep-docker') == None:
        exit_program_early("Cannot locate program 'fmriprep-docker', make sure it is in your PATH.")

    uid = Popen(["id", "-u"], stdout=PIPE).stdout.read().decode("utf-8").strip()
    gid = Popen(["id", "-g"], stdout=PIPE).stdout.read().decode("utf-8").strip()
    cifti_out_res = '91k'

    helper_command = shlex.split(f"""{shutil.which('fmriprep-docker')} 
                                 --user {uid}:{gid}
                                 --participant_label {subject}
                                 --cifti-output {cifti_out_res}
                                 --use-syn-sdc warn
                                 {option_chain}
                                 {bids_path.as_posix()}
                                 {derivs_path.as_posix()}
                                 participant""")
    try:
        print(f"Running fmriprep-docker with the following command: \n  { " ".join(helper_command) } \n")
        with Popen(helper_command, stdout=PIPE) as p:
            while p.poll() == None:
                text = p.stdout.read1().decode("utf-8", "ignore")
                print(text, end="", flush=True)
            if p.poll() != 0:
                raise RuntimeError("'fmriprep-docker' has ended with a non-zero exit code.")
    except RuntimeError as e:
        print(e) 
        exit_program_early("Program 'fmriprep-docker' has run into an error.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog="ocean_proc.py", 
            description="ocean labs adult mri processing"
        )
    parser.add_argument("--subject", "-su", required=True,
                        help="")
    parser.add_argument("--session", "-se", required=True,
                        help="")
    parser.add_argument("--bids_path", "-b", required=True,
                        help="") 
    parser.add_argument("--source_data", "-sd", required=True,
                        help="")
    parser.add_argument("--xml_path", "-x", required=True,
                        help="The path to the xml file for this session")
    parser.add_argument("--bids_config1", "-c1", required=True,
                        help="")
    parser.add_argument("--bids_config2", "-c2", 
                        help="")
    parser.add_argument("--nordic", "-n", action="store_true", required=False,
                        help="")
    parser.add_argument("--derivs_path", "-d", required=True,
                        help="")
    parser.add_argument("--work_dir", "-w", required=True,
                        help="")
    parser.add_argument("--fs_license", "-l", required=True,
                        help="")
    parser.add_argument("--fs-subjects-dir", "-fs", 
                        help="")
    parser.add_argument("--skip_dcm2bids", action="store_true",
                        help="")
    parser.add_argument("--skip_fmap_pairing", action="store_true",
                        help="")
    parser.add_argument("--skip_fmriprep", action="store_true",
                        help="")
    
    args = parser.parse_args()

    ##### Convert raw DICOMs to BIDS structure #####
    if not args.skip_dcm2bids:
        dicom_to_bids(
            subject=args.subject,
            session=args.session,
            source_dir=args.source_data,
            bids_dir=args.bids_path,
            xml_path=args.xml_path,
            bids_config1=args.bids_config1,
            bids_config2=args.bids_config2,
            nordic=args.nordic
        )

    ##### Pair field maps to functional runs #####
    bids_session_dir = f"{args.bids_path}/sub-{args.subject}/ses-{args.session}"

    if not args.skip_fmap_pairing:
        map_fmap_to_func(
            xml_path=args.xml_path, 
            bids_dir_path=bids_session_dir
        )


    ##### Run fMRIPrep #####
    all_opts = dict(args._get_kwargs())
    fmrip_options = {"work_dir", "fs_license", "fs-subjects-dir"}
    fmrip_opt_chain = " ".join([make_option(fo, all_opts[fo]) for fo in fmrip_options])

    if not args.skip_fmriprep:
        run_fmri_prep(
            subject=args.subject, 
            bids_path=Path(args.bids_path),
            derivs_path=Path(args.derivs_path),
            option_chain=fmrip_opt_chain
        )

    print("####### [DONE] Finished all processing, exiting now #######")

    