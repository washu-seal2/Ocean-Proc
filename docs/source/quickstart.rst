Quickstart Guide
================

Prerequisites
-------------

* ``conda``, which can be installed `here <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_. 
* ``docker``, which can be installed `here <https://docs.docker.com/engine/install/>`_. 

Installing fmriprep, fmriprep_wrapper, dcm2bids
-----------------------------------------------

To install the ``fmriprep`` Docker image::

    docker pull nipreps/fmriprep:<latest-version>

``oceanproc`` also uses the ``fmriprep_wrapper`` command, which can be installed to your Conda environment. Ensure your environment is active, then::

    python -m pip install fmriprep-docker

To install ``dcm2bids``::

    conda install -c conda-forge dcm2bids


Ensure you have permissions
---------------------------

Since ``oceanproc`` involves the use of a Docker container with root permissions on whatever computer you are using (local, remote node, etc.), you must ensure that you are added to the groups needed to be able to run said container. Contact your system administrator for details if needed.


Set up your dataset directory
-----------------------------

From this point, I'm assuming you're using some sort of UNIX-flavored shell, either on your local computer or on a remote node. These include Bash, zsh, csh, etc.

First, make a directory titled ``my_bids_dataset``, like so::

    mkdir my_bids_dataset
    cd my_bids_dataset

Then, run the following command::

    dcm2bids_scaffold

This will create the directory structure recommended for working with ``oceanproc``, which should look something like this::

    .
    ├── CHANGES
    ├── code
    ├── dataset_description.json
    ├── derivatives
    ├── participants.json
    ├── participants.tsv
    ├── README
    ├── sourcedata
    └── tmp_dcm2bids
        └── log
            └── scaffold_20240628-163701.log

Here's a small breakdown of what these folders mean, from top to bottom:

* ``code``: contains any dataset specific scripts or config files. ``oceanproc`` typically uses this folder to store dcm2bids config file(s), as well as an optional ``arg_file.json`` that contains settings for ``oceanproc`` to be reused between separate subjects/sessions.
* ``derivatives``: in BIDS format, this contains any derivative of raw data, which will be kept in a directory named ``rawdata`` at this same level. In the context of ``oceanproc``, this directory will contain the outputs of ``fmriprep``. 
* ``sourcedata``: contains the raw data that's collected at the scanner. Data doesn't need to be BIDS compliant in this directory, as it will be converted into BIDS format when ``dcm2bids`` is ran; ``dcm2bids`` will also use the config files kept in ``code`` to create a mapping of names and file locations between the raw DICOM/NIFTI data and BIDS-compliant, renamed NIFTI data.
* ``tmp_dcm2bids``: this is a remnant of the ``dcm2bids_scaffold`` command we just ran. From within this directory, we can remove it like so::

    rm -rf tmp_dcm2bids

Set up the ``sourcedata`` directory
-----------------------------------

The structure of this directory does not matter, as it is only for storing raw data in non-BIDS format, which presumably will contain some PHI (and will not be shared). In both cases, move the top-level directory (detailed in the examples below) into the ``sourcedata`` directory, under any name you'll be able to remember.

DICOM
^^^^^

``oceanproc`` supports raw data both in DICOM and NIFTI format. If your data is in DICOM format, ``oceanproc`` will assume that the data is in the same structure that it is in when downloaded from the CNDA database::

    your_data_directory
    ├── 1
    │   └── DICOM
    │       ├── xxxxxxxxx.x.xx.x.xxxx.x.x.xx.xxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxx-x-x-jxxxmb.dcm
    │       └── x.x.xx.x.xxxx.x.x.xx.xxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxx-x-x-bqomui.dcm
    ├── 10
    │   └── DICOM
    │       └── x.x.xx.x.xxxx.x.x.xx.xxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxx-xx-x-bimxxp.dcm
    ├── 11
    │   └── DICOM
    │       └── x.x.xx.x.xxxx.x.x.xx.xxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxx-xx-x-bimxxp.dcm
    ...

The top-level session directory should contain directories named after each series number, a subdirectory in each of these named 'DICOM', and then all the .dcm files for that series number kept in this directory. 

NIFTI
^^^^^

For NIFTI raw data, just keep all of the .nii (or .nii.gz) and .json files in the same level, as in this example::

    your_data_directory/
    ├── DICOM_AAHScout_xxxxxxxxxxxxxx_2.json
    ├── DICOM_AAHScout_xxxxxxxxxxxxxx_2.nii.gz
    ├── DICOM_AAHScout_xxxxxxxxxxxxxx_30.json
    ├── DICOM_AAHScout_xxxxxxxxxxxxxx_30.nii.gz
    ├── DICOM_AAHScout_xxxxxxxxxxxxxx_31.json
    ├── DICOM_AAHScout_xxxxxxxxxxxxxx_31.nii.gz
    ├── DICOM_AAHScout_xxxxxxxxxxxxxx_32.json
    ├── DICOM_AAHScout_xxxxxxxxxxxxxx_32.nii.gz
    ├── DICOM_AAHScout_xxxxxxxxxxxxxx_33.json
    ...


Building your dcm2bids config file
----------------------------------

A full detailed guide to building ``dcm2bids`` config files can be found `here <https://unfmontreal.github.io/Dcm2Bids/3.1.1/tutorial/first-steps/#how-to-use-this-tutorial>`_. 

This config is needed to determine how to name each series run, and where within the final BIDS hierarchy it will live. This naming convention can be determined by a number of factors, all of which are derived from the .json "sidecar" file containing metadata for the scan. 

If your raw data is in the ``sourcedata`` directory and in DICOM format, the command ``dcm2bids_helper`` can generate a list of these sidecar files with the metadata needed to make the config; if it's already in NIFTI format, you can look through each of these files to get the identifying info that you'll need. 

 | 2024-07-03 Wed 12:38 PM


