Advanced usage tips
===================

Skipping different steps of processing
--------------------------------------

Since ``oceanproc`` is composed of multiple processing steps, we provide the option to skip some of them with these options:

* ``--skip_dcm2bids``: Specify this option if your data is already BIDS-formatted in the ``rawdata`` directory.
* ``--skip_fmap_pairing``: Specify this option if you don't want to perform fieldmap pairing with BOLD runs.
* ``--skip_fmriprep``: Specify this option if you don't want to run fmriprep on the BIDS-formatted raw data
* ``--skip_event_files``: Specify this option if you want to skip the creation of long-formatted event files
