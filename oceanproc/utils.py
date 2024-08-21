import sys


def exit_program_early(msg:str, exit_func=None):
    """
    Exit the program while printing parameter 'msg'. If an exit
    function is sepcified, it will be called before the program
    exits

    :param msg: error message to display.
    :type msg: str
    :param exit_func: function to be called before program exit (no required arguments)
    :type exit_func: function

    """
    print(f"---[ERROR]: {msg} \nExiting the program now...")
    if exit_func and callable(exit_func):
        exit_func()
    sys.exit(1)


def prompt_user_continue(msg:str) -> bool:
    """
    Prompt the user to continue with a custom message.

    :param msg: prompt message to display.
    :type msg: str

    """
    user_continue = input(f"""
        {msg} 
        ---(press 'y' for yes, other input will mean no) 
    """)
    return user_continue.lower() == "y"


def make_option(value, key: str=None, delimeter: str=" ", convert_underscore: bool=False):
    """
    Generate a string, representing an option that gets fed into a subprocess.

    For example, if a key is 'option' and its value is True, the option string it will generate would be:

        --option

    If value is equal to some string 'value', then the string would be:

        --option value

    If value is a list of strings:

        --option value1 value2 ... valuen

    :param key: Name of option to pass into a subprocess, without double hyphen at the beginning.
    :type key: str
    :param value: Value to pass in along with the 'key' param.
    :type value: str or bool or list[str] or None
    :param delimeter: character to separate the key and the value in the option string. Default is a space.
    :type delimeter: str
    :param convert_underscore: flag to indicate that underscores should be replaced with '-'
    :type convert_underscore: bool
    :return: String to pass as an option into a subprocess call.
    :rtype: str
    """
    second_part = None
    if key and convert_underscore:
        key = key.replace("_", "-")
    if not value:
        return ""
    if type(value) == bool and value:
        second_part = " "
    if type(value) == list:
        second_part = f"{delimeter}{' '.join([str(v) for v in value])}"
    else:
        second_part = f"{delimeter}{str(value)}"
    return f"--{key}{second_part}" if key else second_part