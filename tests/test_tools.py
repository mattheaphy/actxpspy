import pytest
from actxps.tools import *

l = ['guac', 'cheese', 'beans', 'rice']

class TestArgMatch():
    
    def test_allowed(self):
        assert arg_match('toppings', 'guac', l) is None
    
    def test_nont_allowed(self):
        with pytest.raises(ValueError, match = '`toppings` must be one of'):
            arg_match('toppings', 'bananas', l)