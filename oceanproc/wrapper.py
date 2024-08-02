import subprocess
from .oceanparse import OceanParser

def check_docker():
    """Verify that docker is installed and the user has permission to
    run docker images.

    Returns
    -------
    -1  Docker can't be found
     0  Docker found, but user can't connect to daemon
     1  Test run OK
    """
    try:
        ret = subprocess.run(['docker', 'version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        from errno import ENOENT

        if e.errno == ENOENT:
            return -1
        raise e
    if ret.stderr.startswith(b'Cannot connect to the Docker daemon.'):
        return 0
    return 1

def get_parser():
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
    session_arguments.add_argument("--source_data", "-sd", required=True,
                        help="The path to the directory containing the raw DICOM files for this subject and session")
    session_arguments.add_argument("--xml_path", "-x", required=True,
                        help="The path to the xml file for this subject and session")
    session_arguments.add_argument("--fs_subjects_dir", "-fs", 
                        help="The path to the directory that contains previous FreeSurfer outputs/derivatives to use for fMRIPrep. If empty, this is the path where new FreeSurfer outputs will be stored.")
    session_arguments.add_argument("--skip_dcm2bids", action="store_true",
                        help="Flag to indicate that dcm2bids does not need to be run for this subject and session")
    session_arguments.add_argument("--skip_fmap_pairing", action="store_true",
                        help="Flag to indicate that the pairing of fieldmaps to BOLD runs does not need to be performed for this subject and session")
    session_arguments.add_argument("--skip_fmriprep", action="store_true",
                        help="Flag to indicate that fMRIPrep does not need to be run for this subject and session")
    session_arguments.add_argument("--skip_event_files", action="store_true",
                        help="Flag to indicate that the making of a long formatted events file is not needed for the subject and session")
    session_arguments.add_argument("--export_args", "-ea", 
                        help="Path to a file to save the current configuration arguments")
    session_arguments.add_argument("--keep_work_dir", action="store_true",
                        help="Flag to stop the deletion of the fMRIPrep working directory")

    config_arguments.add_argument("--bids_path", "-b",  required=True,
                        help="The path to the directory containing the raw nifti data for all subjects, in BIDS format") 
    config_arguments.add_argument("--bids_config", "-c", required=True,
                        help="The path to the dcm2bids config file to use for this subject and session")
    config_arguments.add_argument("--nordic_config", "-n", 
                        help="The path to the second dcm2bids config file to use for this subject and session. This implies that the session contains NORDIC data")
    config_arguments.add_argument("--nifti", action=argparse.BooleanOptionalAction,
                        help="Flag to specify that the source directory contains files of type NIFTI (.nii/.jsons) instead of DICOM")
    config_arguments.add_argument("--anat_only", action='store_true',
                        help="Flag to specify only anatomical images should be processed.")
    config_arguments.add_argument("--fd_spike_threshold", "-fd", type=float, default=0.9,
                        help="framewise displacement threshold (in mm) to determine outlier framee (Default is 0.9).")
    config_arguments.add_argument("--skip_bids_validation", action="store_true",
                        help="Specifies skipping BIDS validation (only enabled for fMRIprep step)")
    config_arguments.add_argument("--leave_workdir", action="store_true",
                        help="Don't clean up working directory after fmriprep has finished. We can do this since we're creating a new working directory for every instance of fmriprep (this option should be used as a debugging tool).")
    config_arguments.add_argument("--derivs_path", "-d", required=True,
                        help="The path to the BIDS formated derivatives directory for this subject")
    config_arguments.add_argument("--work_dir", "-w", required=True,
                        help="The path to the working directory used to store intermediate files")
    config_arguments.add_argument("--fs_license", "-l", required=True,
                        help="The path to the license file for the local installation of FreeSurfer")
    config_arguments.add_argument("--no_tty",
                        help="Run docker without TTY flag -it")
    return parser, config_arguments

def build_docker_cmd(opts):
    ret = subprocess.run(
        ['docker', 'version', '--format', '{{.Server.Version}}'], stdout=subprocess.PIPE
    )
    docker_version = ret.stdout.decode('ascii').strip()

    command = ['docker', 'run', '--rm', '-e', 'DOCKER_VERSION_8395080871=%s' % docker_version]

    if not opts.no_tty:
        if opts.help:
            command.append('-i')
        else:
            command.append('-it')
