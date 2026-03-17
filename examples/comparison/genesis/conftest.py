"""Minimal conftest for running Genesis gradient tests outside the Genesis repo."""
import genesis as gs
import pytest

gs.init(backend=gs.gpu, logging_level="warning")


@pytest.fixture
def show_viewer():
    return False


# Genesis tests use @pytest.mark.parametrize("backend", [gs.cpu, gs.gpu]).
# The backend fixture below captures that value and is available if needed,
# but genesis is already initialised above so we just yield the param.
@pytest.fixture
def backend(request):
    yield getattr(request, "param", gs.gpu)
