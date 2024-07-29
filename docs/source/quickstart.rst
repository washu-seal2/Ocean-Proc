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

You'll also want to create a directory on the same level as ``derivatives``, ``sourcedata``, etc. named ``rawdata``, which can be done like so::

    mkdir rawdata

This directory is for storing your raw data after it's been put into BIDS format.

Set up the ``sourcedata`` directory
-----------------------------------

The structure of this directory does not matter, as it is only for storing raw data in non-BIDS format. In both cases, move the top-level directory (detailed in the examples below) into the ``sourcedata`` directory, under any name you'll be able to remember.

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
    ├── ...._1.json
    ├── ...._1.nii.gz
    ├── ...._2.json
    ├── ...._2.nii.gz
    ├── ...._3.json
    ├── ...._3.nii.gz
    ...
    ├── ...._n.json
    ├── ...._n.nii.gz


Building your dcm2bids config file
----------------------------------

A more detailed guide to building ``dcm2bids`` config files can be found `here <https://unfmontreal.github.io/Dcm2Bids/3.2.0/how-to/create-config-file/>`_. 

The dcm2bids config file is kept in the ``code`` directory at the top level of your BIDS directory, and is typically named something like ``dcm2bids_config.json``. This config is needed to determine how each raw data file will be named, and where they will reside in the BIDS hierarchy. This naming convention can be determined by a number of factors, all of which are derived from the .json "sidecar" file containing metadata for the scan. 

If your raw data is in the ``sourcedata`` directory and in DICOM format, the command ``dcm2bids_helper`` can generate a list of these sidecar files with the metadata needed to make the config. To run this, simply navigate to the directory containing all of the folders with DICOM files (they should be named `1`, `2`, ... `99`), and run::
    
    dcm2bids_helper

from the command line. If your data is already in NIFTI format, you can look through each of these files to get the identifying info that you'll need. 

Here's a brief example of what a mapping of fieldmap files into BIDS format would look like in a config file::

    {
        "descriptions": [
            {
                "datatype": "fmap",
                "suffix": "epi",
                "custom_entities": "dir-AP",
                "criteria": {
                    "SeriesDescription": "SpinEchoFieldMap_AP_2p4mm", 	
                "ImageTypeText": ["ORIGINAL", "PRIMARY", "M", "ND"]
                }
            },
            {
                "datatype": "fmap",
                "suffix": "epi",
                "custom_entities": "dir-PA",
                "criteria": {
                    "SeriesDescription": "SpinEchoFieldMap_PA_2p4mm", 	
                "ImageTypeText": ["ORIGINAL", "PRIMARY", "M", "ND"]
                }
            }
        ]
    } 

Let's break down what these items mean.

* ``"descriptions"``: This top-level field in the .json file will contain a list of mappings from raw NIFTI format into BIDS format.
    * ``"datatype"``: This is a mandatory field, and describes under which subfolder below the session level this series will be contained in. The BIDS v1.2.0 specification defines the following six:
        * ``"anat"``: specified for anatomical images 
        * ``"beh"``: specified for behavioral data
        * ``"dwi"``: specified for diffusion-weighted imaging
        * ``"fmap"``: specified for fieldmaps
        * ``"func"``: specified for BOLD functional data
        * ``"meg"``: specified for magnetoencephalography (MEG)
    * ``"custom_entities"``: This field allows for adding additional information in the BIDS-compliant file name. In the above example, "dir-AP" or "dir-PA" will be added to the fieldmap names to specify which direction they were collected in (anterior -> posterior, or posterior -> anterior)
    * ``"criteria"``: This field allows for only choosing files that include every specified key-value pair in their JSON sidecar file. In the above examples, dcm2bids will only convert fieldmap files into NIFTI format if their "SeriesDescription" value matches either "SpinEchoFieldMap_AP_2p4mm" or "SpinEchoFieldMap_PA_2p4mm", and whose "ImageTypeText" field matches the list of specified image types.

There are many other configuration options available when building your config file; refer to the link at the top of this section for more details.

Running oceanproc
-----------------






