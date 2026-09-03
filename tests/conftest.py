import sys
from pathlib import Path

# Make `backend.*` importable when pytest is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from backend.config.settings import REPO_ROOT
from backend.experiment.experiment_loader import load_experiment


@pytest.fixture
def sample_experiment():
    return load_experiment(REPO_ROOT / "experiments" / "bas_sample_001.json")
