# uv workflow

First, make sure you have `uv` installed in your Python environment. Instructions for your specific system can be found [here](https://docs.astral.sh/uv/getting-started/installation/).

After cloning the repository, navigate to the top-level directory of the repo (this should contain `uv.lock` and `pyproject.toml`), and run:

```
uv sync
```

This will create (or update, if it exists) a virtual environment under a directory named `.venv`. To activate this for running entry-point scripts defined in `pyproject.toml` for testing, for example, simply activate the environment like:

```
. .venv/bin/activate
```

to get access to those entrypoint scripts. To deactivate the enviornment, run:

```
deactivate
```

If you want to create a build of the project, run:

```
uv build
```

### adding packages

`uv add <package-name>`

### running pytest, sphinx, or other dev dependencies

Make sure that when running any dev tool, you prepend it with `uv run`. For example, to run
pytest: 

`uv run pytest [other-args...]`.

# uploading to pypi

First ensure that you have a `dist/` directory containing the wheel and compressed package files, and that the version of the package you're wanting to upload is different than what's already up on pypi (this can be changed in `pyproject.toml`. 

Also, make sure you have a valid API token for uploading to the oceanproc project on pypi. You can make a `.pypirc` in your home (`~`) directory, with the following contents:

```
[pypi]
username = __token__
password = <the token value, including the `pypi-` prefix>
```

This file will allow you to upload to pypi without needing to reenter the API token each time.

Then, to upload the new version:

`uv run twine upload dist/*`
