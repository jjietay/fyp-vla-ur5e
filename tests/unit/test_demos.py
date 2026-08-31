"""Unit tests for the demonstration capture path: the buffer and the HDF5 file.

This is the code that stands between fifteen hours of lab time and a dataset. A
bug here is not found until export, or worse until training, and the only fix is
to record everything again.

`DemoRecorder` never touches a robot and `hdf5_store` never touches anything but
a file, so both are fully testable with no hardware. There was no reason for
these to be uncovered other than that nobody had written them.
"""

from __future__ import annotations

import numpy as np
import pytest

from fyp.architecture_b.demos.hdf5_store import episode_paths, load_episode, save_snapshots
from fyp.architecture_b.demos.recorder import DemoRecorder

FRAME = (4, 6, 3)


def a_frame(i: int = 0) -> dict:
    """It takes an index and gives you one plausible timestep of robot state."""
    return {
        "joint_positions": np.full(6, 0.1 * i, dtype=float),
        "tcp_pose": np.array([0.1 * i, 0.2, 0.3, 3.142, 0.0, 0.0], dtype=float),
        "gripper_state": i % 2,
        "image": np.full(FRAME, i, dtype=np.uint8),
    }


def a_recording(n: int = 5) -> DemoRecorder:
    r = DemoRecorder()
    r.start_episode()
    for i in range(n):
        r.record(**a_frame(i), timestamp=0.05 * i)
    return r


# ------------------------------------------------------------------ buffer

def test_recording_before_starting_is_refused():
    """
    Without this the first frames of an episode land in whatever the buffer held
    from the previous one, and the episode is silently contaminated.
    """
    with pytest.raises(RuntimeError, match="start_episode"):
        DemoRecorder().record(**a_frame())


def test_starting_again_discards_the_previous_episode():
    r = a_recording(3)
    r.start_episode()
    assert len(r) == 0


def test_frames_are_kept_in_order(): 
    r = a_recording(5)
    assert [f.timestamp for f in r.frames] == pytest.approx([0.0, 0.05, 0.10, 0.15, 0.20])


def test_an_explicit_timestamp_is_used_verbatim():
    """
    The hardware recorder polls from a thread and knows when the sample was
    really taken. If the buffer overwrote that with its own clock, the timestamps
    would describe when the frame was filed rather than when it happened, and the
    inferred control rate would be wrong.
    """
    r = DemoRecorder()
    r.start_episode()
    r.record(**a_frame(), timestamp=42.5)
    assert r.frames[0].timestamp == 42.5


def test_an_omitted_timestamp_is_measured_from_the_start():
    r = DemoRecorder()
    r.start_episode()
    r.record(**a_frame())
    assert 0.0 <= r.frames[0].timestamp < 1.0


# -------------------------------------------------------------------- file

def test_a_saved_episode_reads_back_unchanged(tmp_path):
    r = a_recording(5)
    path = tmp_path / "ep_001.h5"
    r.save_episode(path)

    data = load_episode(path)
    assert set(data) == {"timestamps", "joint_positions", "tcp_poses",
                         "gripper_states", "images"}
    assert len(data["timestamps"]) == 5
    np.testing.assert_allclose(data["tcp_poses"][2], a_frame(2)["tcp_pose"])
    np.testing.assert_array_equal(data["images"][3], a_frame(3)["image"])


def test_every_dataset_has_the_same_length(tmp_path):
    """
    The export validator rejects an episode whose datasets disagree, and it is
    right to. This checks the writer never produces one.
    """
    path = tmp_path / "ep_001.h5"
    a_recording(7).save_episode(path)

    data = load_episode(path)
    assert len({len(v) for v in data.values()}) == 1


def test_gripper_states_survive_as_integers(tmp_path):
    path = tmp_path / "ep_001.h5"
    a_recording(4).save_episode(path)
    assert list(load_episode(path)["gripper_states"]) == [0, 1, 0, 1]


# ------------------------------------------------------------------ paths

def test_episodes_sort_by_name_across_both_extensions(tmp_path):
    """
    Regression test. The old version returned every .h5 before every .hdf5
    regardless of name, so mixed extensions silently reordered the dataset:
    frames stayed intact, episode boundaries moved, and nothing raised. You would
    only notice as a model that will not converge.
    """
    for name in ("ep_003.h5", "ep_001.hdf5", "ep_002.h5", "ep_004.hdf5"):
        (tmp_path / name).touch()

    assert [p.name for p in episode_paths(tmp_path)] == [
        "ep_001.hdf5", "ep_002.h5", "ep_003.h5", "ep_004.hdf5"]


def test_an_empty_folder_gives_an_empty_list(tmp_path):
    assert episode_paths(tmp_path) == []


def test_unrelated_files_are_ignored(tmp_path):
    (tmp_path / "ep_001.h5").touch()
    (tmp_path / "notes.md").touch()
    (tmp_path / "ep_001.mp4").touch()
    assert [p.name for p in episode_paths(tmp_path)] == ["ep_001.h5"]
