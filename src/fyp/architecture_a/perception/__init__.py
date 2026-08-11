"""Stages 2 and 3: what is on the table, and where it is in 3D.

    detector.py     OWLv2 open-vocabulary detection, raw and unfiltered
    filters.py      thresholding and NMS, applied separately so the detector
                    stays a pure model call
    pixel_to_3d.py  pinhole back-projection, CameraIntrinsics, depth_at
    localiser.py    detections plus depth -> LocatedObject with camera XYZ

Open-vocabulary detection is what gives Architecture A its generalisation
story: a new object is a new text query, not a new dataset.

Torch dependency: `detector.py` only. Everything else runs in the FYP venv.
"""
