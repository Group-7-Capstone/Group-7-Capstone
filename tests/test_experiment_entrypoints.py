"""The moved experiment imports from arbitrary working directories without execution."""
from pathlib import Path
import importlib


def test_builders_import_from_outside_experiment(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    experiment = root/'experiments/camera-ticket-trends'
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(experiment))
    for name in ['build_deliverable', 'build_shared_analysis', 'build_road_trends',
                 'build_trend_walkthrough', 'shared_presentation']:
        module = importlib.import_module(name)
        if hasattr(module, 'ROOT'):
            assert module.ROOT == root
    from validation.camera_shared_trends import describe_matrix, SCENARIOS
    assert callable(describe_matrix) and 'balanced_52' in SCENARIOS
    assert not list(tmp_path.iterdir())
