import subprocess
from pathlib import Path
import os
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
root_dir = Path(os.environ['DATA1'])

@app.route('/')
def oceanproc():
    return "<h1>Hello, world!</h1>"

@app.route('/files/')
@app.route('/files/<path:subpath>')
def get_filesystem(subpath=""):
    app.logger.debug(f"path: {subpath}")
    full_path = Path(os.path.realpath(f"{root_dir.as_posix()}/{subpath}"))
    if not full_path.is_dir():
        return render_template('filesystem_fail.html', nonexistant_path=full_path.as_posix())
    dirs, files = [], []
    for f in full_path.iterdir():
        file_obj = {}
        file_obj["isdir"] = True if f.is_dir() else False
        file_obj["basename"] = f.name
        file_obj["realpath"] = f.resolve().as_posix()
        file_obj["pathfromroot"] = f.relative_to(root_dir)
        dirs.append(file_obj) if f.is_dir() else files.append(file_obj)
    dirs = sorted(dirs, key=lambda f: f['basename'])
    files = sorted(files, key=lambda f: f['basename'])
    file_list = dirs + files
    if full_path != root_dir:
        app.logger.info("here")
        file_list.insert(0, {"isdir": True, "basename": "..", "realpath": full_path.as_posix(), "pathfromroot": re.sub(r'(.*)/.*', r'\1', full_path)})
    return render_template('filesystem.html', 
                           cur_dir=full_path,
                           file_list=file_list)

@app.route('/api/get_filesystem/', methods=['GET'])
@app.route('/api/get_filesystem/<path:subpath>', methods=['GET'])
def get_filesystem_json(subpath=""):
    full_path = Path(os.path.realpath(f"{root_dir.as_posix()}/{subpath}"))
    dirs, files = [], []
    for f in full_path.iterdir():
        file_obj = {}
        file_obj["isdir"] = True if f.is_dir() else False
        file_obj["basename"] = f.name
        file_obj["realpath"] = f.resolve().as_posix()
        file_obj["pathfromroot"] = f.relative_to(root_dir).as_posix()
        dirs.append(file_obj) if f.is_dir() else files.append(file_obj)
    dirs = sorted(dirs, key=lambda f: f['basename'])
    files = sorted(files, key=lambda f: f['basename'])
    file_list = dirs + files
    if full_path != root_dir:
        app.logger.info("here")
        file_list.insert(0, {'isdir': True,
                             'basename': "..",
                             'realpath': full_path.resolve().as_posix(),
                             'pathfromroot': full_path.parents[0].relative_to(root_dir).as_posix()})
    return jsonify({'cur_dir': full_path.as_posix(),
                    'file_list': file_list}), 200


@app.route('/api/get_parser_args/', methods=['GET'])
def get_parser_args():
    from .get_parser_json import generate_json_from_parser_options
    parser_args = generate_json_from_parser_options()
    return jsonify(parser_args), 200


@app.route('/formtest/', methods=['GET'])
def formtest():
    return render_template('oceanproc_form.html'), 200

        
    
      

