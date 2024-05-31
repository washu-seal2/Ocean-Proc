import pytest
import os
from ..group_series import *

@pytest.fixture
def xml_path():
    return f"{os.path.dirname(__file__)}/assets/test.xml"

def test_get_locals_from_xml(xml_path):
    localizers = get_locals_from_xml(xml_path)
    assert isinstance(localizers, list)
    assert localizers == sorted(localizers)

