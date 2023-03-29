from actxps.expose import *
from actxps.datasets import *
import pytest

toy_census = load_toy_census()

class TestExposeInit():
    
    def test_bad_expo_length(self):
        with pytest.raises(ValueError, match = 'must be one of'):
            ExposedDF(toy_census, '2022-12-31', expo_length='quantum')