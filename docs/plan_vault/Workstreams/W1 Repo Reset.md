---
status: done
needs_lab: false
week: 1
---

# W1 Repo Reset

Housekeeping that has to happen before real data exists, because each item silently corrupts something downstream if left.

- [x] split the package into `shared/`, `architecture_a/` and `architecture_b/`, imports rewritten, history preserved
- [x] add `euler_to_rotvec` and `euler_to_R` to `shared/helpers/rotations.py`, with round trip tests against the existing `rotvec_to_euler`
- [x] fix `shared/helpers/config.py::get_config(path)`, which ignores `path` after the first call because of a module level singleton, so it looks configurable and is not
- [x] fix `architecture_b/demos/hdf5_store.py::episode_paths`, which returns sorted `.h5` then sorted `.hdf5`, so mixed extensions give silently wrong episode order
- [x] make `export_lerobot.py` fail loudly on a truncated or malformed episode instead of exporting it
- [x] make `export_lerobot.py` assert the reported fps matches the configured record rate

## Why these and not other cleanups

Each one is a **silent** failure. The episode ordering bug corrupts a dataset without raising anything, and you would find it after training. The rotation helpers block [[W2 Architecture A Software]] outright, since nothing that commands an orientation can be written without them.

## Done 11 Aug 2026

All items closed. Two things surfaced that were not on the list:

* `R_to_rotvec` lost about seven digits of precision as the angle approached pi, because it divided by `sin(theta)`. That is not an edge case here, since the default tool orientation is the gripper pointing down at theta almost exactly pi. Replaced with a quaternion pivot method, now exact to 1e-15
* controller tests are marked `integration` and skip when nothing answers on port 30004, so `pytest tests` is green on a machine with no robot
