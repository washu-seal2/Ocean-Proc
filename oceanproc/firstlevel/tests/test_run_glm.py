from ..run_glm import *

def test_make_option():
    assert make_option(True) == " "
    assert make_option(True, key="myopt") == "--myopt "
    assert make_option(['1','2','3']) == " 1 2 3"
    assert make_option("hello") == " hello"
    assert make_option("hello", key="myopt") == "--myopt hello"

