import sys


def exit_program_early(msg:str):
    """
    Exit the program while printing parameter 'msg'.

    :param msg: error message to display.
    :type param: str

    """
    print(f"---[ERROR]: {msg} \nExiting the program now...")
    sys.exit(1)


def prompt_user_continue(msg:str) -> bool:
    """
    Prompt the user to continue with a custom message.

    :param msg: prompt message to display.
    :type msg: str

    """
    user_continue = input(f"--- {msg} (press 'y' for yes, other input will mean no) ---")
    return user_continue.lower() == "y"