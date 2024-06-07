# poetry workflow

When first cloning the repository, make sure that the version of Python you're using is >=3.10. 

Then, run the following:

```
python3 -m venv .venv
. .venv/bin/activate
pip install poetry==1.8.3
poetry install
```

### adding packages

Run `poetry add` with the .venv environment activated to add any new packages to the environment.

### building

Run `poetry build` with the .venv environment activated to create a new build; this will generate a 
`dist/` directory at the top level that will contain the .whl file and .tar.gz containing the 
source code.

### running pytest, sphinx, or other dev dependencies

Make sure that when running any dev tool, you prepend it with `poetry run`. For example, to run
pytest: 

`poetry run pytest [other-args...]`.

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

`poetry run twine upload dist/*`
