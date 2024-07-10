import ast
import asttokens
from html import escape
import codecs
import json
import os
import re
import re
import subprocess
import sys
from textwrap import dedent

def _create_argument_dicts(path: str, parser_group_names: list[dict]):
    def get_opt(node):
        toklist = list(script_ast.get_tokens(node))
        optname = [re.sub(r'\"--(.*)\"', r'\1', t.string) for t in toklist if 
                   t.type == 3 and 
                   ('--' in t.string or '-' not in t.string)][0]
        equalindices = [idx for idx in range(len(toklist)) if toklist[idx].string == "="]
        optsettingkeys = [toklist[idx-1].string for idx in equalindices]
        optsettingvals = []
        for startidx in equalindices:
            optsettingval = ""
            idx = startidx + 1
            while toklist[idx].string != "," and idx < len(toklist) - 1:
                optsettingval += re.sub(r'^[\"\'](.*)[\"\']$', r'\1', toklist[idx].string)
                idx += 1
            optsettingvals.append(optsettingval)
        option = {k: v for (k, v) in zip(optsettingkeys, optsettingvals)}
        return option
    arguments = []
    with open(path) as f:
        script_ast = asttokens.ASTTokens(f.read(), parse=True)
        for node in ast.walk(script_ast.tree):
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id in parser_group_names and
                isinstance(node.func.attr, str) and
                node.func.attr == 'add_argument'):
                    arguments.append(get_opt(node))
    return arguments

def generate_json_from_parser_options(config_path: str = 'config.json') -> list[dict]:
    """
    Returns a list of Python dicts, which will be interpreted in JSON format by a React component for generating an HTML form corresponding to command-line arguments.

    :return: A Python dict containing two fields: 'option', the name of the argparse option, and 'html_type', a string containing the corresponding HTML <input> type field. 
    :rtype: list[dict]
    """
    with open(config_path) as f:
        config = json.load(f)
    options_script_path = config["options_script_path"]
    if not os.path.isfile(options_script_path):
        raise RuntimeError(f"ERROR: script at {options_script_path} does not exist.")
    parser_group_names = config["parser_group_names"]
    argument_dicts = _create_argument_dicts(options_script_path, parser_group_names)
    return argument_dicts
