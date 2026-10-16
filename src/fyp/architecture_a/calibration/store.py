"""One place that owns the T_base_cam file: where it is, how it is replaced, who may validate it.

Architecture A only. Every script that writes the calibration and the pipeline
that reads it go through `CalibrationStore`, so the rules below live once.

- A replaced file is moved aside as `T_base_cam.<time>.json`, never overwritten.
  Every earlier result must be able to name the calibration it was produced with.
- Every replace sets `validated` to False. Only `mark_validated` (the reach test,
  `scripts/check_calibration.py`) sets it True. A fit residual cannot see a wrong
  `ball_offset_tool_m` or a table-height bias, so a fit must never vouch for itself.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from fyp.shared.helpers.config import get_config, resolve

FILENAME = "T_base_cam.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class CalibrationStore:
    """
    It takes the config (or an explicit file path) and gives you the current
    T_base_cam with its provenance, and one safe way to replace it.

    `path` overrides the config location for the scripts that take `--out` or
    `--calibration`. The archive goes next to whichever file that is.
    """

    def __init__(self, cfg: dict | None = None, path: str | Path | None = None):
        if path is not None:
            self.path = Path(path)
        else:
            cfg = cfg or get_config()
            self.path = resolve(cfg["paths"]["calibration_dir"]) / FILENAME

    def exists(self) -> bool:
        return self.path.is_file()

    def current(self) -> tuple[np.ndarray, dict]:
        """
        It takes nothing and gives you (T_base_cam, meta) as stored, or raises
        FileNotFoundError / ValueError when there is no usable file.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"no calibration at {self.path}. Run the hand-eye procedure first; "
                "without T_base_cam the pipeline cannot convert a detection into a "
                "reachable target."
            )
        data = json.loads(self.path.read_text())
        T = np.asarray(data["T_base_cam"], dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"{self.path} holds a {T.shape} array, expected (4, 4)")
        return T, dict(data.get("meta", {}))

    def replace(self, T: np.ndarray, step: str, meta: dict, *,
                restart: bool = False) -> Path | None:
        """
        It takes the new transform, the name of the step that made it and that
        step's facts, and gives you the path the old file was moved to (None if
        there was no old file).

        By default the old meta is kept and the step's facts are added under
        `meta[step]`, so the file shows the whole chain (ball fit, then level,
        then place-and-look). `restart=True` is for a step that makes a new
        calibration from scratch (ball fit, measured mount): the old meta is
        dropped and the facts become the top level of the new meta.
        """
        T = np.asarray(T, dtype=float)
        archived = self._archive()
        facts = {"written_at": _now(), **meta}
        if archived is not None:
            facts["from_file"] = archived.name

        if restart:
            new_meta = facts
        else:
            new_meta = self._old_meta(archived)
            # validated_at belongs to the transform that was just replaced.
            new_meta.pop("validated_at", None)
            new_meta[step] = facts
        new_meta["validated"] = False
        self._write(T, new_meta)
        return archived

    def mark_validated(self, result: dict) -> None:
        """
        It takes the reach test's numbers and records them in the file, leaving
        the transform alone. This is the only way `validated` becomes True.
        """
        T, meta = self.current()
        meta.update(result)
        meta["validated"] = True
        meta["validated_at"] = _now()
        self._write(T, meta)

    # ----------------------------------------------------------------- private

    def _archive(self) -> Path | None:
        if not self.exists():
            return None
        stamp = datetime.fromtimestamp(self.path.stat().st_mtime).strftime("%Y%m%dT%H%M%S")
        archived = self.path.with_name(f"{self.path.stem}.{stamp}{self.path.suffix}")
        self.path.rename(archived)
        return archived

    @staticmethod
    def _old_meta(archived: Path | None) -> dict:
        # A damaged old file must not block writing a good new one.
        if archived is None:
            return {}
        try:
            return dict(json.loads(archived.read_text()).get("meta", {}))
        except (ValueError, OSError):
            return {}

    def _write(self, T: np.ndarray, meta: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"T_base_cam": T.tolist(), "meta": meta}, indent=2))
