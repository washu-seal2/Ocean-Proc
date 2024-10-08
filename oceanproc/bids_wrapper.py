#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import OrderedDict
from glob import glob
from pathlib import Path
import json
import os
import re
import shlex
import shutil
import subprocess
from textwrap import dedent
import xml.etree.ElementTree as et
from .utils import exit_program_early, prompt_user_continue, prepare_subprocess_logging, debug_logging, log_linebreak, flags
import logging

logger = logging.getLogger(__name__)

@debug_logging
def remove_unusable_runs(xml_file:Path, bids_data_path:Path, subject:str):
    """
    Will remove unusable scans from list of scans after dcm2bids has run.

    :param xml_file: Path to XML file containing quality information generated by XNAT.
    :type xml_file: pathlib.Path
    :param bids_data_path: Path to directory containing BIDS-compliant data for the given subject.
    :type bids_data_path: pathlib.Path
    :param subject: Subject ID used in BIDS-compliant data (for example, if 'sub-5000', subject is '5000').
    :type subject: str

    """
    log_linebreak()
    logger.info("####### Removing the scans marked 'unusable' #######\n")

    if not xml_file.exists():
        exit_program_early(f"Path {str(xml_file)} does not exist.")
        
    tree = et.parse(xml_file)
    prefix = "{" + str(tree.getroot()).split("{")[-1].split("}")[0] + "}"
    scan_element_list = list(tree.iter(f"{prefix}scans"))
    
    if len(scan_element_list) != 1:
        exit_program_early(f"Error parsing the xml file provided. Found none or more than one scan groups")
    
    scans = scan_element_list[0]
    quality_pairs = {int(s.get("ID")) : s.find(f"{prefix}quality").text
                     for s in scans}
    
    if len(quality_pairs) == 0:
        exit_program_early("Could not find scan quality information in the given xml file.") 

    logger.debug(f"scan quality information: {quality_pairs}")
    
    json_paths = sorted(list(p for p in (bids_data_path / f"sub-{subject}/").rglob("*.json"))) # if re.search(json_re, p.as_posix()) != None)
    nii_paths = sorted(list(p for p in (bids_data_path / f"sub-{subject}/").rglob("*.nii.gz"))) # if re.search(json_re, p.as_posix()) != None)

    if len(json_paths) == 0 or len(nii_paths) == 0:
        exit_program_early("Could not find Nifti or JSON sidecar files in the bids directory.")

    if len(json_paths) != len(nii_paths):
        exit_program_early("Unequal amount of NIFTI and JSON files found")
 
    for p_json, p_nii in zip(json_paths, nii_paths):
        j = json.load(p_json.open()) 
        if quality_pairs[j["SeriesNumber"]] == "unusable":
            logger.info(f"  Removing series {j['SeriesNumber']}: NIFTI:{p_nii}, JSON:{p_json}")
            os.remove(p_json) 
            os.remove(p_nii) 


@debug_logging
def run_dcm2bids(subject:str, 
                 session:str,
                 source_dir:Path, 
                 bids_output_dir:Path, 
                 config_file:Path, 
                 nordic_config:Path=None,
                 nifti:bool=False):
    """
    Run dcm2bids with a given set of parameters.

    :param subject: Subject name (ex. 'sub-5000', subject would be '5000')
    :type subject: str
    :param session: Session name (ex. 'ses-01', session would be '01')
    :type session: str
    :param source_dir: Path to 'sourcedata' directory (or wherever DICOM data is kept).
    :type source_dir: pathlib.Path
    :param bids_output_dir: Path to the bids directory to store the newly made NIFTI files
    :type bids_output_dir: pathlib.Path
    :param config_file: Path to dcm2bids config file, which maps raw sourcedata to BIDS-compliant counterpart
    :type config_file: pathlib.Path
    :param nordic_config: Path to second dcm2bids config file, needed for additional post processing that one BIDS config file can't handle.
    :type nordic_config: pathlib.Path
    :param nifti: Specify that the soure directory contains NIFTI files instead of DICOM
    :type nifti: bool
    :raise RuntimeError: If dcm2bids exits with a non-zero exit code.
    """

    for p in [source_dir, bids_output_dir, config_file]:
        if not p.exists():
            exit_program_early(f"Path {str(p)} does not exist.")

    if shutil.which('dcm2bids') == None:
            exit_program_early("Cannot locate program 'dcm2bids', make sure it is in your PATH.")
    
    tmp_path = bids_output_dir / f"tmp_dcm2bids/sub-{subject}_ses-{session}"
    
    def clean_up(quiet=False):
        try:
            logger.debug(f"removing the temporary directory used by dcm2bids: {tmp_path}")
            shutil.rmtree(tmp_path)
        except Exception:
            if not quiet:
                logger.warning(f"There was a problem deleting the temporary directory at {tmp_path}")
    
    if (path_that_exists := bids_output_dir/f"sub-{subject}/ses-{session}").exists():
        ans = prompt_user_continue(dedent(f"""
                                    A raw data bids path for this subject and session already exists. 
                                    Would you like to delete its contents and rerun dcm2bids? If not,
                                    dcm2bids will be skipped.
                                          """))
        if ans:
            logger.debug("removing the old BIDS raw data directory and its contents")
            shutil.rmtree(path_that_exists)
        else:
            return
        
    nifti_path = None    
    if not nifti:
        clean_up(quiet=True)
        run_dcm2niix(source_dir=source_dir, 
                     tmp_nifti_dir=tmp_path)
        nifti_path = tmp_path
    else:
        nifti_path = source_dir
    helper_command = shlex.split(f"""{shutil.which('dcm2bids')} 
                                 --bids_validate 
                                 --skip_dcm2niix
                                 -d {str(nifti_path)} 
                                 -p {subject} 
                                 -s {session} 
                                 -c {str(config_file)} 
                                 -o {str(bids_output_dir)}
                                 """)
    try:
        log_linebreak()
        logger.info("####### Running first round of Dcm2Bids ########\n")
        prepare_subprocess_logging(logger)
        with subprocess.Popen(helper_command, stdout=subprocess.PIPE) as p:    
            while p.poll() == None:
                for line in p.stdout:
                    logger.info(line.decode("utf-8", "ignore"))
            prepare_subprocess_logging(logger, stop=True)
            p.kill()
            if p.poll() != 0:
                raise RuntimeError("'dcm2bids' has ended with a non-zero exit code.")
            
        if nordic_config:
            if not nordic_config.exists():
                exit_program_early(f"Path {nordic_config} does not exist.")

            nordic_run_command = shlex.split(f"""{shutil.which('dcm2bids')} 
                                            --bids_validate
                                            --skip_dcm2niix
                                            -d {str(nifti_path)} 
                                            -p {subject}
                                            -s {session}
                                            -c {str(nordic_config)}
                                            -o {str(bids_output_dir)}
                                            """)
            log_linebreak()
            logger.info("####### Running second round of Dcm2Bids ########\n")
            prepare_subprocess_logging(logger)
            with subprocess.Popen(nordic_run_command, stdout=subprocess.PIPE) as p:
                while p.poll() == None:
                    for line in p.stdout:
                        logger.info(line.decode("utf-8", "ignore"))
                prepare_subprocess_logging(logger, stop=True)
                p.kill()
                if p.poll() != 0:
                    raise RuntimeError("'dcm2bids' has ended with a non-zero exit code.")
                
            # Clean up NORDIC files
            if not flags.debug:
                separate_nordic_files = glob(f"{str(bids_output_dir)}/sub-{subject}/ses-{session}/func/*_part-*")
                logger.debug(f"removing the old nordic files that are not needed after mag-phase combination :\n  {separate_nordic_files}")
                for f in separate_nordic_files:
                    os.remove(f)

    except RuntimeError or subprocess.CalledProcessError as e:
        prepare_subprocess_logging(logger, stop=True)
        logger.exception(e, stack_info=True)
        exit_program_early("Problem running 'dcm2bids'.", None if flags.debug else clean_up)
    if not flags.debug:
        clean_up()


@debug_logging
def run_dcm2niix(source_dir:Path, 
                 tmp_nifti_dir:Path,
                 clean_up_func=None):
    """
    Run dcm2niix with the given input and output directories.

    :param source_dir: Path to 'sourcedata' directory (or wherever DICOM data is kept).
    :type source_dir: pathlib.Path
    :param tmp_nifti_dir: Path to the directory to store the newly made NIFTI files
    :type tmp_nifti_dir: pathlib.Path
    """
    
    if not source_dir.exists():
        exit_program_early(f"Path {source_dir} does not exist.")
    elif shutil.which('dcm2niix') == None:
        exit_program_early("Cannot locate program 'dcm2niix', make sure it is in your PATH.")

    if not tmp_nifti_dir.exists():
        tmp_nifti_dir.mkdir(parents=True)
    
    helper_command = shlex.split(f"""{shutil.which('dcm2niix')} 
                                -b y
                                -ba y
                                -z y
                                -f %3s_%f_%p_%t
                                -o {str(tmp_nifti_dir)}
                                {str(source_dir)}
                                """)
    try: 
        log_linebreak()
        logger.info("####### Converting DICOM files into NIFTI #######\n")
        prepare_subprocess_logging(logger)
        with subprocess.Popen(helper_command, stdout=subprocess.PIPE) as p:
            while p.poll() == None:
                for line in p.stdout:
                    logger.info(line.decode("utf-8", "ignore"))
            prepare_subprocess_logging(logger, stop=True)
            p.kill()
            if p.poll() != 0:
                raise RuntimeError("'dcm2bniix' has ended with a non-zero exit code.")
    except RuntimeError or subprocess.CalledProcessError as e:
        prepare_subprocess_logging(logger, stop=True)
        logger.exception(e, stack_info=True)
        exit_program_early("Problem running 'dcm2niix'.", 
                           exit_func=clean_up_func if clean_up_func else None)
        
    # Delete or move extra files from short runs
    files_to_remove = list(tmp_nifti_dir.glob("*a.nii.gz")) + list(tmp_nifti_dir.glob("*a.json"))
    if flags.debug:
        unused_files_dir = tmp_nifti_dir.parent / f"{tmp_nifti_dir.name}_unused"
        unused_files_dir.mkdir(exist_ok=True)
        logger.debug(f"moving some unused files and files created from shortened runs to directory {unused_files_dir}")
        for f in files_to_remove:
            shutil.move(f.resolve(), (unused_files_dir/f.name).resolve())
    else:
        logger.info(f"removing some unused files and files created from shortened runs :\n  {[str(f) for f in files_to_remove]}")
        for f in files_to_remove:
            os.remove(f)
    
    
@debug_logging
def dicom_to_bids(subject:str, 
                  session:str, 
                  source_dir:Path, 
                  bids_dir:Path, 
                  xml_path:Path, 
                  bids_config:Path,
                  nordic_config:Path=None,
                  nifti=False):
    
    """
    Facilitates the conversion of DICOM data into NIFTI data in BIDS format, and the removal of data marked 'unusable'.

    :param subject: Subject name (ex. 'sub-5000', subject would be '5000')
    :type subject: str
    :param session: Session name (ex. 'ses-01', session would be '01')
    :type session: str
    :param source_dir: Path to 'sourcedata' directory (or wherever DICOM data is kept).
    :type source_dir: pathlib.Path
    :param bids_dir: Path to the bids directory to store the newly made NIFTI files
    :type bids_dir: pathlib.Path
    :param bids_config: Path to dcm2bids config file, which maps raw sourcedata to BIDS-compliant counterpart
    :type bids_config: pathlib.Path
    :param nordic_config: Path to second dcm2bids config file, needed for additional post processing if NORDIC data that one BIDS config file can't handle.
    :type nordic_config: pathlib.Path
    :param nifti: Specify that the soure directory contains NIFTI files instead of DICOM
    :type nifti: bool
    """

    run_dcm2bids(subject=subject, 
                 session=session,
                 source_dir=source_dir, 
                 bids_output_dir=bids_dir, 
                 config_file=bids_config, 
                 nordic_config=nordic_config, 
                 nifti=nifti)
    
    remove_unusable_runs(xml_file=xml_path, 
                         bids_data_path=bids_dir, 
                         subject=subject)


if __name__ == "__main__":
    parser = ArgumentParser(prog="bids_wrapper.py",
                                    description="wrapper script for dcm2bids",
                                    epilog="WIP")
    parser.add_argument("-su", "--subject", required=True, 
                        help="Subject ID")
    parser.add_argument("-se","--session", required=True, 
                        help="Session ID")
    parser.add_argument("-sd", "--source_data", type=Path, required=True,
                        help="Path to directory containing this session's DICOM files")
    parser.add_argument("-b", "--bids_path", type=Path, required=True, 
                        help="Path to the bids directory to store the newly made NIFTI files")
    parser.add_argument("-x", "--xml_path", type=Path, required=True, 
                        help="Path to this session's XML file")
    parser.add_argument("-c", "--bids_config", type=Path, required=True, 
                        help="dcm2bids config json file")
    parser.add_argument("-n", "--nordic_config", type=Path,
                        help="Second dcm2bids config json file used for NORDIC processing")
    parser.add_argument("--nifti", action='store_true', 
                        help="Flag to specify that the source directory contains files of type NIFTI (.nii/.jsons) instead of DICOM")
    args = parser.parse_args()
    
    dicom_to_bids(subject=args.subject,
                  session=args.session,
                  source_dir=args.source_data,
                  bids_dir=args.bids_path,
                  xml_path=args.xml_path,
                  bids_config=args.bids_config,
                  nordic_config=args.nordic_config,
                  nifti=args.nifti)
