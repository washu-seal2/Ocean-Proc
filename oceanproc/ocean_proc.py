#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import logging
import datetime
from .bids_wrapper import dicom_to_bids
from .group_series import map_fmap_to_func
from .events_long import create_events_and_confounds
from .utils import exit_program_early, prompt_user_continue, make_option, prepare_subprocess_logging, default_log_format, add_file_handler, export_args_to_file, flags, debug_logging
from .oceanparse import OceanParser
import shlex
import shutil
from subprocess import Popen, PIPE
from textwrap import dedent

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(stream=sys.stdout)],
                    format=default_log_format)
logger = logging.getLogger() 

@debug_logging(this_logger=logger)
def make_work_directory(dir_path:Path, subject:str, session:str) -> Path:
    dir_to_make = dir_path / f"sub-{subject}_ses-{session}"
    if dir_to_make.exists():
        want_to_delete = prompt_user_continue(dedent("""
            A work directory already exists for this subject and session. 
            Would you like to delete its contents and start fresh?
            """))
        if want_to_delete:
            logger.debug("removing the old working directory and its contents")
            shutil.rmtree(dir_to_make)
        else:
            return dir_to_make
    dir_to_make.mkdir()
    logger.info(f"creating a new working directory at the path: {dir_to_make}")
    return dir_to_make


@debug_logging(this_logger=logger)
def run_fmri_prep(subject:str,
                  bids_path:Path,
                  derivs_path:Path,
                  option_chain:str,
                  remove_work_folder:Path=None):
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
    :param remove_work_folder: Path to the working directory that will be deleted upon completion or error (default None)
    :type remove_work_folder: str
    :raise RuntimeError: If fmriprep throws an error, or exits with a non-zero exit code.
    """
    clean_up = lambda : shutil.rmtree(remove_work_folder) if remove_work_folder else None

    logger.info("####### Starting fMRIPrep #######")
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
                                 {str(bids_path)}
                                 {str(derivs_path)}
                                 participant""")
    try:
        command_str = " ".join(helper_command)
        logger.info(f"Running fmriprep-docker with the following command: \n  {command_str} \n")
        prepare_subprocess_logging(logger)
        with Popen(helper_command, stdout=PIPE) as p:
            while p.poll() == None:
                for line in p.stdout:
                    logger.info(line.decode("utf-8", "ignore"))
            prepare_subprocess_logging(logger, stop=True)
            p.kill()
            if p.poll() != 0:
                raise RuntimeError("'fmriprep-docker' has ended with a non-zero exit code.")
    except RuntimeError as e:
        prepare_subprocess_logging(logger, stop=True)
        logger.exception(e, stack_info=True) 
        exit_program_early("Program 'fmriprep-docker' has run into an error.", 
                           None if flags.debug else clean_up)
    if not flags.debug:
        clean_up()
    


def main():
    parser = OceanParser(
            prog="oceanproc", 
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
    session_arguments.add_argument("--source_data", "-sd", type=Path, required=True,
                        help="The path to the directory containing the raw DICOM files for this subject and session")
    session_arguments.add_argument("--xml_path", "-x", type=Path, required=True,
                        help="The path to the xml file for this subject and session")
    session_arguments.add_argument("--fs_subjects_dir", "-fs", type=Path,
                        help="The path to the directory that contains previous FreeSurfer outputs/derivatives to use for fMRIPrep. If empty, this is the path where new FreeSurfer outputs will be stored.")
    session_arguments.add_argument("--skip_dcm2bids", action="store_true",
                        help="Flag to indicate that dcm2bids does not need to be run for this subject and session")
    session_arguments.add_argument("--skip_fmap_pairing", action="store_true",
                        help="Flag to indicate that the pairing of fieldmaps to BOLD runs does not need to be performed for this subject and session")
    session_arguments.add_argument("--skip_fmriprep", action="store_true",
                        help="Flag to indicate that fMRIPrep does not need to be run for this subject and session")
    session_arguments.add_argument("--skip_event_files", action="store_true",
                        help="Flag to indicate that the making of a long formatted events file is not needed for the subject and session")
    session_arguments.add_argument("--export_args", "-ea", type=Path,
                        help="Path to a file to save the current configuration arguments")
    session_arguments.add_argument("--keep_work_dir", action="store_true",
                        help="Flag to stop the deletion of the fMRIPrep working directory")
    session_arguments.add_argument("--debug", action="store_true",
                        help="Flag to run the program in debug mode for more verbose logging")

    config_arguments.add_argument("--bids_path", "-b", type=Path, required=True,
                        help="The path to the directory containing the raw nifti data for all subjects, in BIDS format") 
    config_arguments.add_argument("--derivs_path", "-d", type=Path, required=True,
                        help="The path to the BIDS formated derivatives directory for this subject")
    config_arguments.add_argument("--derivs_subfolder", "-ds", default="fmriprep",
                        help="The subfolder in the derivatives directory where bids style outputs should be stored. The default is 'fmriprep'.")
    config_arguments.add_argument("--bids_config", "-c", type=Path, required=True,
                        help="The path to the dcm2bids config file to use for this subject and session")
    config_arguments.add_argument("--nordic_config", "-n", type=Path,
                        help="The path to the second dcm2bids config file to use for this subject and session. This implies that the session contains NORDIC data")
    config_arguments.add_argument("--nifti", action=argparse.BooleanOptionalAction,
                        help="Flag to specify that the source directory contains files of type NIFTI (.nii/.jsons) instead of DICOM")
    config_arguments.add_argument("--anat_only", action='store_true',
                        help="Flag to specify only anatomical images should be processed.")
    config_arguments.add_argument("--fd_spike_threshold", "-fd", type=float, default=0.9,
                        help="framewise displacement threshold (in mm) to determine outlier framee (Default is 0.9).")
    config_arguments.add_argument("--skip_bids_validation", action="store_true",
                        help="Specifies skipping BIDS validation (only enabled for fMRIprep step)")
    config_arguments.add_argument("--work_dir", "-w", type=Path, required=True,
                        help="The path to the working directory used to store intermediate files")
    config_arguments.add_argument("--fs_license", "-l", type=Path, required=True,
                        help="The path to the license file for the local installation of FreeSurfer")
    config_arguments.add_argument("--fmriprep_version", "-fv", default="23.1.4", dest="image",
                        help="The version of fmriprep to use. The default is 23.1.4. It is reccomended that an entire study use the same version.")
    args = parser.parse_args()

    try:
        assert args.derivs_path.is_dir(), "Derivatives directory must exist but it cannot be found"
        assert args.bids_path.is_dir(), "Raw Bids directory must exist but it cannot be found"
        assert args.source_data.is_dir(), "Source data directory must exist but it cannot be found"
        assert args.work_dir.is_dir(), "Work directroy must exist but it cannot be found"
    except AssertionError as e:
        logger.exception(e)
        exit_program_early(e)

    ##### Export the current configuration arguments to a file #####
    if args.export_args:
        try:
            assert args.export_args.parent.exists() and args.export_args.suffix, "Argument export path must be a file path in a directory that exists"
            logger.info(f"####### Exporting Configuration Arguments to: '{args.export_args}' #######")
            export_args_to_file(args, config_arguments, args.export_args)
        except Exception as e:
            logger.exception(e)
            exit_program_early(e)

    args.image = f"nipreps/fmriprep:{args.image}"

    args.derivs_path = args.derivs_path / args.derivs_subfolder

    log_dir = args.derivs_path / f"sub-{args.subject}/log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sub-{args.subject}_ses-{args.session}_oceanproc_desc-{datetime.datetime.now().strftime('%m-%d-%y_%I-%M%p')}.log"
    add_file_handler(logger, log_path)

    if args.debug:
        flags.debug = True
        logger.setLevel(logging.DEBUG)

    logger.info("Starting oceanproc...")
    logger.info(f"Log will be stored at {log_path}")

    # log the input arguments
    for k,v in (dict(args._get_kwargs())).items():
        logger.info(f" {k} : {v}")

    args.work_dir = make_work_directory(dir_path=args.work_dir, 
                                        subject=args.subject, 
                                        session=args.session)

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
        )

    ##### Pair field maps to functional runs #####
    bids_session_dir = args.bids_path / f"sub-{args.subject}/ses-{args.session}"

    if not args.anat_only and not args.skip_fmap_pairing:
        map_fmap_to_func(
            xml_path=args.xml_path, 
            bids_dir_path=bids_session_dir
        )


    ##### Run fMRIPrep #####
    all_opts = dict(args._get_kwargs())

    fmrip_options = {"work_dir", 
                     "fs_license", 
                     "fs_subjects_dir", 
                     "skip_bids_validation", 
                     "fd_spike_threshold", 
                     "anat_only",
                     "image"}
    fmrip_opt_chain = " ".join([make_option(all_opts[fo], key=fo, delimeter="=", convert_underscore=True) for fo in fmrip_options if fo in all_opts])

    if not args.skip_fmriprep:
        run_fmri_prep(
            subject=args.subject, 
            bids_path=args.bids_path,
            derivs_path=args.derivs_path,
            option_chain=fmrip_opt_chain,
            remove_work_folder=None if args.keep_work_dir else args.work_dir
        )

    
    ##### Create long formatted event files #####
    if not args.skip_event_files:
        create_events_and_confounds(
            bids_path=args.bids_path,
            derivs_path=args.derivs_path,
            sub=args.subject,
            ses=args.session,
            fd_thresh=args.fd_spike_threshold
        )
    
    logger.info("\n")
    logger.info("####### [DONE] Finished all processing, exiting now #######")

    
if __name__ == "__main__":
    main()
