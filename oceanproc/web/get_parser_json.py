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

def _find_add_argument_strings(path: str, parser_group_names: list[str]) -> list[str]:
    calls_list = []
    with open(path) as f:
        script_ast = asttokens.ASTTokens(f.read(), parse=True)
        for node in ast.walk(script_ast.tree):
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id in parser_group_names and
                isinstance(node.func.attr, str) and
                node.func.attr == 'add_argument'):
                    calls_list.append(script_ast.get_text(node))
    return calls_list
     
def _extract_action_from_arg_string(arg_string: str) -> str:
    try: # Has a defined option
        action = re.search(r'action=[\'\"]([\w\d_]+)[\'\"]', arg_string).group(1)
        return action
    except AttributeError: # Has the default option 'store'
        return 'store'

def _extract_optname_from_arg_string(arg_string: str) -> str:
    optname = re.search(r'add_argument\([\"\']--([\w\d_]+)', arg_string).group(1)
    return optname

def _build_html_dict(option: str, action: str) -> dict:
    if action == 'store':
        return {"option": option, "html_type": "text"}
    elif action == 'store_true':
        return {"option": option, "html_type": "checkbox"}
    else:
        return {"option": "INVALID", "html_type": "INVALID"} # unimplemented option

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
    add_argument_strings = _find_add_argument_strings(options_script_path, parser_group_names)
    options_to_actions = {_extract_optname_from_arg_string(a): _extract_action_from_arg_string(a) for a in add_argument_strings}
    html_dicts = [] 
    for opt, act in options_to_actions.items():
        html_dicts.append(_build_html_dict(opt, act))
    return html_dicts
