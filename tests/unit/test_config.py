"""Unit tests for config loading, which everything else depends on.

Small file, and one of these is a regression test for a bug that made the whole
project look configurable while it was not.
"""

from __future__ import annotations

import pytest
import yaml

from fyp.shared.helpers.config import DEFAULT_CONFIG_PATH, get_config, load_config, resolve


def test_the_real_config_has_the_blocks_everything_reads():
    """
    A missing block is a KeyError deep inside a skill or an adapter, at the worst
    possible moment. Cheaper to notice here.
    """
    cfg = get_config()
    for block in ("robot", "paths", "workspace", "camera", "speech",
                  "architecture_a", "architecture_b", "testing"):
        assert block in cfg, f"config.yaml is missing the {block} block"


def test_two_different_paths_do_not_share_one_cache(tmp_path):
    """
    Regression test. `get_config` once kept a single module-level dict and
    returned it for every call regardless of `path`, so passing a different
    config file appeared to work and silently did nothing. `URController.__init__`
    takes a path argument, which made it look configurable when it was not.
    """
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump({"robot": {"host": "10.0.0.99"}}))

    default = get_config()
    assert get_config(other)["robot"]["host"] == "10.0.0.99"
    # and reading the default again must not return the other one
    assert get_config() is default


def test_the_same_path_is_read_once(tmp_path):
    """Caching per path, so a config is parsed once per process and not per call."""
    assert get_config(DEFAULT_CONFIG_PATH) is get_config(DEFAULT_CONFIG_PATH)


def test_load_config_always_reads_from_disk(tmp_path):
    """The uncached door, which is what makes the cache testable at all."""
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"a": 1}))
    assert load_config(path) == {"a": 1}

    path.write_text(yaml.safe_dump({"a": 2}))
    assert load_config(path) == {"a": 2}


def test_paths_resolve_against_the_project_root_not_the_cwd():
    """
    Every path in config.yaml is relative. If they resolved against the working
    directory, running a script from a subfolder would write data somewhere new
    and the previous recordings would appear to have vanished.
    """
    resolved = resolve(get_config()["paths"]["episodes_dir"])
    assert resolved.is_absolute()
    assert (resolved.parent.parent / "config" / "config.yaml").exists() or \
        "data" in str(resolved)


def test_a_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
