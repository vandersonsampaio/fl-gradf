import pytest

from tests.fixtures.mock_data import make_synthetic_participants


@pytest.fixture
def synthetic_participants():
    return make_synthetic_participants()
