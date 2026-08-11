---
status: in progress
needs_lab: false
week: 1
---

# W1 Repo Reset

Housekeeping that has to happen before real data exists, because each item silently corrupts something downstream if left.

- [x] quarantine the simulation into `simulation/`, imports rewritten, history preserved
- [ ] add `euler_to_rotvec` and `euler_to_R` to `shared/helpers/rotations.py`, with round trip tests against the existing `rotvec_to_euler`
- [ ] fix `shared/helpers/config.py::get_config(path)`, which ignores `path` after the first call because of a module level singleton, so it looks configurable and is not
- [ ] fix `architecture_b/demos/hdf5_store.py::episode_paths`, which returns sorted `.h5` then sorted `.hdf5`, so mixed extensions give silently wrong episode order
- [ ] make `export_lerobot.py` fail loudly on a truncated or malformed episode instead of exporting it
- [ ] make `export_lerobot.py` assert the reported fps matches the configured record rate

## Why these and not other cleanups

Each one is a **silent** failure. The episode ordering bug corrupts a dataset without raising anything, and you would find it after training. The rotation helpers block [[W2 Architecture A Software]] outright, since nothing that commands an orientation can be written without them.
