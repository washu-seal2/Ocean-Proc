# syntax=docker/dockerfile:1

# TODO

# Change the entrypoint to run a version of oceanproc that's NOT the wrapper

# set environment vars
FROM nipreps/fmriprep:latest
ENV OCEANPROC_ROOT=/opt/oceanproc
COPY . ${OCEANPROC_ROOT}/src
WORKDIR ${OCEANPROC_ROOT}/src
SHELL ["/bin/bash", "-c"]

# install oceanproc
RUN \
	python -m venv .venv && \
	. .venv/bin/activate && \
	pip install poetry && \
	poetry lock && \
	poetry install && \
	poetry build && \
	deactivate && \
	pushd dist && \
	pip install *.tar.gz

ENTRYPOINT ["oceanproc"]

# TODO: 
# - automatically replace any paths pointing to nordicbash.sh with a path from within the container

	
		
