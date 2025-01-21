Usage
=====

Command line options
--------------------

Here's the output of ``oceanproc -h``:

.. code-block:: text 

    usage: oceanproc [-h] --subject SUBJECT --session SESSION --source_data
                     SOURCE_DATA --xml_path XML_PATH [--skip_dcm2bids]
                     [--skip_fmap_pairing] [--skip_fmriprep] [--skip_event_files]
                     [--export_args EXPORT_ARGS] [--keep_work_dir] [--debug]
                     [--fmap_pairing_file FMAP_PAIRING_FILE] --bids_path BIDS_PATH
                     --derivs_path DERIVS_PATH
                     [--derivs_subfolder DERIVS_SUBFOLDER] --bids_config
                     BIDS_CONFIG [--nordic_config NORDIC_CONFIG]
                     [--nifti | --no-nifti] [--anat_only]
                     [--fd_spike_threshold FD_SPIKE_THRESHOLD]
                     [--skip_bids_validation] [--fs_subjects_dir FS_SUBJECTS_DIR]
                     --work_dir WORK_DIR --fs_license FS_LICENSE
                     [--fmriprep_version IMAGE]

    Ocean Labs adult MRI preprocessing

    options:
      -h, --help            show this help message and exit

    Session Specific:
      --subject SUBJECT, -su SUBJECT
                            The identifier of the subject to preprocess
      --session SESSION, -se SESSION
                            The identifier of the session to preprocess
      --source_data SOURCE_DATA, -sd SOURCE_DATA
                            The path to the directory containing the raw DICOM
                            files for this subject and session
      --xml_path XML_PATH, -x XML_PATH
                            The path to the xml file for this subject and session
      --skip_dcm2bids       Flag to indicate that dcm2bids does not need to be run
                            for this subject and session
      --skip_fmap_pairing   Flag to indicate that the pairing of fieldmaps to BOLD
                            runs does not need to be performed for this subject
                            and session
      --skip_fmriprep       Flag to indicate that fMRIPrep does not need to be run
                            for this subject and session
      --skip_event_files    Flag to indicate that the making of a long formatted
                            events file is not needed for the subject and session
      --export_args EXPORT_ARGS, -ea EXPORT_ARGS
                            Path to a file to save the current configuration
                            arguments
      --keep_work_dir       Flag to stop the deletion of the fMRIPrep working
                            directory
      --debug               Flag to run the program in debug mode for more verbose
                            logging
      --fmap_pairing_file FMAP_PAIRING_FILE
                            Path to JSON containing info on how to pair fieldmaps
                            to BOLD runs.

    Configuration Arguments:
      These arguments are saved to a file if the '--export_args' option is used

      --bids_path BIDS_PATH, -b BIDS_PATH
                            The path to the directory containing the raw nifti
                            data for all subjects, in BIDS format
      --derivs_path DERIVS_PATH, -d DERIVS_PATH
                            The path to the BIDS formated derivatives directory
                            for this subject
      --derivs_subfolder DERIVS_SUBFOLDER, -ds DERIVS_SUBFOLDER
                            The subfolder in the derivatives directory where bids
                            style outputs should be stored. The default is
                            'fmriprep'.
      --bids_config BIDS_CONFIG, -c BIDS_CONFIG
                            The path to the dcm2bids config file to use for this
                            subject and session
      --nordic_config NORDIC_CONFIG, -n NORDIC_CONFIG
                            The path to the second dcm2bids config file to use for
                            this subject and session. This implies that the
                            session contains NORDIC data
      --nifti, --no-nifti   Flag to specify that the source directory contains
                            files of type NIFTI (.nii/.jsons) instead of DICOM
      --anat_only           Flag to specify only anatomical images should be
                            processed.
      --fd_spike_threshold FD_SPIKE_THRESHOLD, -fd FD_SPIKE_THRESHOLD
                            framewise displacement threshold (in mm) to determine
                            outlier framee (Default is 0.9).
      --skip_bids_validation
                            Specifies skipping BIDS validation (only enabled for
                            fMRIprep step)
      --fs_subjects_dir FS_SUBJECTS_DIR, -fs FS_SUBJECTS_DIR
                            The path to the directory that contains previous
                            FreeSurfer outputs/derivatives to use for fMRIPrep. If
                            empty, this is the path where new FreeSurfer outputs
                            will be stored.
      --work_dir WORK_DIR, -w WORK_DIR
                            The path to the working directory used to store
                            intermediate files
      --fs_license FS_LICENSE, -l FS_LICENSE
                            The path to the license file for the local
                            installation of FreeSurfer
      --fmriprep_version IMAGE, -fv IMAGE
                            The version of fmriprep to use. The default is 23.1.4.
                            It is reccomended that an entire study use the same
                            version.

    An arguments file can be accepted with @FILEPATH




Specifying which fieldmaps to use with which BOLD runs
------------------------------------------------------

Sometimes, depending on the quality of fieldmaps collected at the scanner, you'll want to specify which fieldmaps should be paired with which BOLD runs for distortion correction. This can be done by specifying a pairing file, a JSON containing this information, using the ``--fmap_pairing_file`` option.. Below is an example of this pairing file:

.. code-block:: json

    {
        "pairings": [
            {
                "fmap": "run-04",
                "func": [
                    "task-reward1",
                    "task-reward2",
                    "task-reward3",
                    "task-reward4"
                ]
            }
        ]
    }


The "pairings" list here should containg a list of objects containing an "fmap" field with one item, and a "func" field with a list of strings.

A JSON file with the above contents fed into `oceanproc`, for example, will add all BOLD images containing the substrings "task-reward1", "task-reward2", "task-reward3", and "task-reward4" in their filename to the "IntendedFor" field in all the fieldmap JSON files containing the substring "run-04".
