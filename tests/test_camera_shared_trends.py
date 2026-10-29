import numpy as np
import pandas as pd
import pytest

from preprocessing.camera_shared_trends import DEFAULTS, aligned_panels, normalize_baseline, shape_features, descriptive_rows
from validation.camera_shared_trends import bootstrap_mean, select_shape_model


def fixture_frames():
    summary = pd.DataFrame({"camera_id": ["a", "b"], "inferred_activation_date": ["2020-12-30", "2021-01-04"]})
    weekly = pd.DataFrame({"camera_id": ["a", "a", "a", "b", "b"],
                          "week_start": pd.to_datetime(["2020-12-28", "2021-01-04", "2021-01-11", "2021-01-04", "2021-01-11"]),
                          "observed_ticket_count": [2, 0, 5, 3, 2],
                          "is_complete_week": [False, True, False, False, True]})
    return summary, weekly


def test_anchor_cross_year_partial_missing_and_zero():
    summary, weekly = fixture_frames()
    meta, primary, admin, calendar = aligned_panels(summary, weekly, {"all_sources_complete": True})
    assert meta.loc["a", "anchor"] == pd.Timestamp("2021-01-04")
    assert primary.loc["a", 1] == 0
    assert np.isnan(primary.loc["a", 2])
    assert np.isnan(primary.loc["b", 1])
    assert primary.loc["b", 2] == 2  # Missing first week does not move the anchor.
    assert admin.loc["a", 2] == 5  # Complete administrative week includes the last partial week.
    assert admin.loc["a", 3] == 0
    assert np.isnan(calendar.loc["b", pd.Timestamp("2021-01-04")])


def test_incomplete_sources_cannot_authorize_zero_extension():
    summary, weekly = fixture_frames()
    _, primary, admin, _ = aligned_panels(summary, weekly, {"all_sources_complete": False})
    pd.testing.assert_frame_equal(primary, admin)


def test_duplicate_week_fails_instead_of_averaging():
    summary, weekly = fixture_frames()
    with pytest.raises(ValueError):
        aligned_panels(summary, pd.concat([weekly, weekly.iloc[[0]]]), {"all_sources_complete": True})


def test_early_coverage_and_cutoff_are_missing():
    summary = pd.DataFrame({"camera_id": ["a"], "inferred_activation_date": ["2016-06-27"]})
    weekly = pd.DataFrame({"camera_id": ["a"], "week_start": pd.to_datetime(["2016-06-27"]), "observed_ticket_count": [1], "is_complete_week": [False]})
    _, _, admin, _ = aligned_panels(summary, weekly, {"all_sources_complete": True}, {**DEFAULTS, "cutoff": "2016-07-13"})
    assert np.isnan(admin.loc["a", 1])
    assert admin.loc["a", 2] == 0
    assert np.isnan(admin.loc["a", 3])


def test_normalization_zero_missing_baselines_and_equal_weights():
    frame = pd.DataFrame([[10, 20, 30], [0, 0, 2], [1, np.nan, 3], [100, 200, 300]], columns=[1, 2, 3])
    indexed = normalize_baseline(frame, [1, 2])
    assert indexed.iloc[[1, 2]].isna().all().all()
    assert indexed.loc[0, 3] == indexed.loc[3, 3] == 200
    assert indexed.loc[0, [1, 2]].mean() == pytest.approx(100)
    assert descriptive_rows(frame).iloc[0]["mean"] == 27.75


def test_bootstrap_preserves_whole_history_and_is_reproducible():
    matrix = np.array([[1, 2], [3, 6], [7, 14]], dtype=float)
    ci = bootstrap_mean(matrix, 200, 42)
    assert np.allclose(ci[:, 1], ci[:, 0] * 2)
    assert np.array_equal(ci, bootstrap_mean(matrix, 200, 42))
    assert np.isnan(bootstrap_mean(np.array([[1, 2]]))).all()
    sparse = bootstrap_mean(np.array([[1, np.nan], [2, 3]]))
    assert np.isnan(sparse[:, 1]).all()


def test_shape_features_constant_profiles_and_block_means():
    frame = pd.DataFrame([np.ones(52), np.arange(52), np.arange(52)[::-1]], index=["flat", "up", "down"])
    features, flat = shape_features(frame)
    assert list(flat) == ["flat"]
    assert features.shape == (2, 13)
    assert np.allclose(features.mean(axis=1), 0, atol=1e-10)
    assert np.allclose(features.std(axis=1, ddof=0), 1)
    assert features.loc["up", 1] < features.loc["up", 13]


def test_clustering_selection_is_reproducible_on_separated_shapes():
    rng = np.random.default_rng(8)
    features = pd.DataFrame(np.concatenate([rng.normal(-3, .1, (30, 13)), rng.normal(3, .1, (30, 13))]))
    config = {**DEFAULTS, "k_values": [2, 3], "stability_repeats": 3}
    model, diagnostics = select_shape_model(features, config)
    assert model.n_clusters == 2
    assert diagnostics.selected.sum() == 1
    assert diagnostics[diagnostics.selected].stability_ari.iloc[0] == 1


def test_no_model_when_quality_thresholds_fail():
    features = pd.DataFrame(np.random.default_rng(1).normal(size=(50, 13)))
    model, diagnostics = select_shape_model(features, {**DEFAULTS, "minimum_silhouette": 1.0, "stability_repeats": 2, "k_values": [2]})
    assert model is None
    assert not diagnostics.selected.any()
