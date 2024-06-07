#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from bids_wrapper import dicom_to_bids, exit_program_early
from group_series import map_fmap_to_func
from oceanparse import OceanParser
import shlex
import shutil
from subprocess import Popen, PIPE
import json


def make_option(key, value, delimeter=" "):
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
    first_part = f"--{key.replace('_', '-')}{delimeter}"
    if value == None:
        return ""
    elif type(value) == bool and value:
        return first_part[:-1]
    elif type(value) == list:
        return first_part + delimeter.join(value)
    elif type(value) == str:
        return first_part + value


def run_fmri_prep(subject:str,
                  bids_path:Path,
                  derivs_path:Path,
                  option_chain:str):
    """
    Run fmriprep with parameters.

    :param subject: Name of subject (ex. if path contains 'sub-5000', subject='5000')
    :type subject: str
    :param bids_path: Path to BIDS-compliant raw data folder.
    :type bids_path: pathlib.Path
    :param derivs_path: Path to BIDS-compliant derivatives folder.
    :type derivs_path: pathlib.Path
    :param option_chain: String containing generated list of options built by make_option().
    :type option_chain: str
    :raise RuntimeError: If fmriprep throws an error, or exits with a non-zero exit code.
    """
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
                                 --participant-label={subject}
                                 --cifti-output={cifti_out_res}
                                 --use-syn-sdc=warn
                                 {option_chain}
                                 {bids_path.as_posix()}
                                 {derivs_path.as_posix()}
                                 participant""")
    try:
        command_str = " ".join(helper_command)
        print(f"Running fmriprep-docker with the following command: \n  {command_str} \n")
        with Popen(helper_command, stdout=PIPE) as p:
            while p.poll() == None:
                text = p.stdout.read1().decode("utf-8", "ignore")
                print(text, end="", flush=True)
            if p.poll() != 0:
                raise RuntimeError("'fmriprep-docker' has ended with a non-zero exit code.")
    except RuntimeError as e:
        print(e) 
        exit_program_early("Program 'fmriprep-docker' has run into an error.")


def main():
    parser = OceanParser(
            prog="ocean_proc.py", 
            description="Ocean Labs adult MRI preprocessing",
            exit_on_error=False,
            fromfile_prefix_chars="@",
            epilog="An arguments file can be accepted with @FILEPATH"
        )
    session_arguments = parser.add_argument_group("Session Specific")
    config_arguments = parser.add_argument_group("Configuration Arguments", "These arguments are saved to a file if the '--export_args' option is used")

    session_arguments.add_argument("--subject", "-su", required=True,
                        help="The identifier of the subject to preprocess")
    session_arguments.add_argument("--session", "-se", required=True,
                        help="The identifier of the session to preprocess")
    config_arguments.add_argument("--bids_path", "-b",  required=True,
                        help="The path to the directory containing the raw nifti data for all subjects, in BIDS format") 
    session_arguments.add_argument("--source_data", "-sd", required=True,
                        help="The path to the directory containing the raw DICOM files for this subject and session")
    session_arguments.add_argument("--xml_path", "-x", required=True,
                        help="The path to the xml file for this subject and session")
    config_arguments.add_argument("--bids_config", "-c", required=True,
                        help="The path to the dcm2bids config file to use for this subject and session")
    config_arguments.add_argument("--nordic_config", "-n", 
                        help="The path to the second dcm2bids config file to use for this subject and session. This implies that the session contains NORDIC data")
    config_arguments.add_argument("--nifti", action="store_true",
                        help="Flag to specify that the source directory contains files of type NIFTI (.nii/.jsons) instead of DICOM")
    # config_arguments.add_argument("--nordic", "-n", action="store_true", required=False,
    #                     help="Flag to indicate that this session contains nordic data. The '--bids_config2' option should also be specified")
    config_arguments.add_argument("--derivs_path", "-d", required=True,
                        help="The path to the BIDS formated derivatives directory for this subject")
    config_arguments.add_argument("--work_dir", "-w", required=True,
                        help="The path to the working directory used to store intermediate files")
    config_arguments.add_argument("--fs_license", "-l", required=True,
                        help="The path to the license file for the local installation of FreeSurfer")
    session_arguments.add_argument("--fs_subjects_dir", "-fs", 
                        help="The path to the directory that contains previous FreeSurfer outputs/derivatives to use for fMRIPrep. If empty, this is the path where new FreeSurfer outputs will be stored.")
    session_arguments.add_argument("--skip_dcm2bids", action="store_true",
                        help="Flag to indicate that dcm2bids does not need to be run for this subject and session")
    session_arguments.add_argument("--skip_fmap_pairing", action="store_true",
                        help="Flag to indicate that the pairing of fieldmaps to BOLD runs does not need to be performed for this subject and session")
    session_arguments.add_argument("--skip_fmriprep", action="store_true",
                        help="Flag to indicate that fMRIPrep does not need to be run for this subject and session")
    session_arguments.add_argument("--export_args", "-ea", 
                        help="Path to a file to save the current configuration arguments")

    args = parser.parse_args()


    ##### Export the current configuration arguments to a file #####
    if args.export_args:
        print(f"####### Exporting Configuration Arguments to: '{args.export_args}' #######")
        all_opts = dict(args._get_kwargs())
        opts_to_save = dict()
        for a in config_arguments._group_actions:
            if all_opts[a.dest]:
                if type(all_opts[a.dest]) == bool:
                    opts_to_save[a.option_strings[0]] = ""
                    continue
                opts_to_save[a.option_strings[0]] = all_opts[a.dest]
        with open(args.export_args, "w") as f:
            if args.export_args.endswith(".json"):
                f.write(json.dumps(opts_to_save))
            else:
                for k,v in opts_to_save.items():
                    f.write(make_option(k,v)+"\n")


    ##### Convert raw DICOMs to BIDS structure #####
    if not args.skip_dcm2bids:
        dicom_to_bids(
            subject=args.subject,
            session=args.session,
            source_dir=args.source_data,
            bids_dir=args.bids_path,
            xml_path=args.xml_path,
            bids_config=args.bids_config,
            nordic_config=args.nordic_config,
            nifti=args.nifti
            # nordic=args.nordic
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
    fmrip_options = {"work_dir", "fs_license", "fs_subjects_dir"}
    fmrip_opt_chain = " ".join([make_option(fo, all_opts[fo], "=") for fo in fmrip_options if fo in all_opts])

    if not args.skip_fmriprep:
        run_fmri_prep(
            subject=args.subject, 
            bids_path=Path(args.bids_path),
            derivs_path=Path(args.derivs_path),
            option_chain=fmrip_opt_chain
        )

    print("####### [DONE] Finished all processing, exiting now #######")

    
if __name__ == "__main__":
    main()
