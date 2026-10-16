"""CalibrationStore: archive on replace, meta chain, validated reset, round trip."""
from __future__ import annotations

import json

import numpy as np
import pytest

from fyp.architecture_a.calibration.store import CalibrationStore
from fyp.shared.helpers.rotations import euler_to_R


def a_transform() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = euler_to_R(np.deg2rad([10.0, -20.0, 30.0]))
    T[:3, 3] = [0.3, -0.2, 0.9]
    return T


@pytest.fixture
def store(tmp_path):
    return CalibrationStore(path=tmp_path / "calib" / "T_base_cam.json")


def test_current_returns_what_replace_wrote(store):
    store.replace(a_transform(), "ball_fit", {"method": "test"}, restart=True)
    T, meta = store.current()
    np.testing.assert_allclose(T, a_transform(), atol=1e-12)
    assert meta["method"] == "test"


def test_first_write_creates_the_directory_and_archives_nothing(store):
    assert store.replace(a_transform(), "ball_fit", {}, restart=True) is None
    assert store.exists()


def test_config_sets_the_path(tmp_path):
    s = CalibrationStore({"paths": {"calibration_dir": str(tmp_path)}})
    assert s.path == tmp_path / "T_base_cam.json"


def test_missing_file_is_refused_with_the_path(store):
    with pytest.raises(FileNotFoundError, match="T_base_cam.json"):
        store.current()


def test_a_file_that_is_not_a_transform_is_refused(store):
    store.path.parent.mkdir(parents=True)
    store.path.write_text('{"T_base_cam": [[1, 0], [0, 1]]}')
    with pytest.raises(ValueError, match=r"\(2, 2\)"):
        store.current()


def test_replace_moves_the_old_file_aside_unchanged(store):
    old = a_transform()
    store.replace(old, "ball_fit", {}, restart=True)
    archived = store.replace(np.eye(4), "levelled_to_table", {})

    assert archived is not None and archived.parent == store.path.parent
    assert archived.name != store.path.name
    np.testing.assert_allclose(
        np.asarray(json.loads(archived.read_text())["T_base_cam"]), old, atol=1e-12)
    np.testing.assert_allclose(store.current()[0], np.eye(4))


def test_chained_replace_keeps_the_earlier_steps_and_names_its_source(store):
    store.replace(a_transform(), "ball_fit", {"method": "ball", "fit_rms_mm": 2.0}, restart=True)
    archived = store.replace(np.eye(4), "levelled_to_table", {"tilt_before_deg": 0.5})

    _, meta = store.current()
    assert meta["method"] == "ball"
    assert meta["levelled_to_table"]["tilt_before_deg"] == 0.5
    assert meta["levelled_to_table"]["from_file"] == archived.name


def test_restart_drops_the_old_meta(store):
    store.replace(a_transform(), "ball_fit", {"method": "ball"}, restart=True)
    store.replace(np.eye(4), "levelled_to_table", {})
    store.replace(a_transform(), "ball_fit", {"method": "ball again"}, restart=True)

    _, meta = store.current()
    assert meta["method"] == "ball again"
    assert "levelled_to_table" not in meta


def test_replace_resets_validated(store):
    store.replace(a_transform(), "ball_fit", {}, restart=True)
    store.mark_validated({"rms_mm": 4.0})
    assert store.current()[1]["validated"] is True

    store.replace(np.eye(4), "place_and_look", {})
    meta = store.current()[1]
    assert meta["validated"] is False
    assert "validated_at" not in meta


def test_a_step_cannot_set_validated_itself(store):
    store.replace(a_transform(), "ball_fit", {}, restart=True)
    store.replace(np.eye(4), "levelled_to_table", {"validated": True})
    assert store.current()[1]["validated"] is False


def test_mark_validated_records_the_numbers_and_keeps_the_transform(store):
    store.replace(a_transform(), "ball_fit", {"method": "ball"}, restart=True)
    store.mark_validated({"rms_mm": 4.0, "n_points": 5})

    T, meta = store.current()
    np.testing.assert_allclose(T, a_transform(), atol=1e-12)
    assert meta["validated"] is True and meta["rms_mm"] == 4.0
    assert meta["method"] == "ball" and "validated_at" in meta


def test_a_damaged_old_file_does_not_block_a_replace(store):
    store.path.parent.mkdir(parents=True)
    store.path.write_text("not json")
    store.replace(a_transform(), "levelled_to_table", {})
    assert store.current()[1]["validated"] is False
