import sys


def exit_program_early(msg:str, exit_func:function=None):
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