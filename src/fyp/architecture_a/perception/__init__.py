"""Stages 2 and 3: what is on the table, and where it is in 3D.

    detector.py     OWLv2 open-vocabulary detection, raw and unfiltered
    filters.py      thresholding and NMS, applied separately so the detector
                    stays a pure model call
    pixel_to_3d.py  pinhole back-projection, CameraIntrinsics, depth_at
    segmenter.py    SAM: one pixel mask per detector box (mask_ray only)
    scene.py        `Perception.see`: detect, filter, localise in one call, plus the
                    floor lift and the in-reach / out-of-reach split. The only
                    entry point the pipeline and the scripts use
    localiser.py    detections plus depth -> LocatedObject with base-frame XYZ.
                    `localise()` picks the method from config: mask_ray (SAM
                    mask, ray cut at the object's centre height) or box_depth

Open-vocabulary detection is what gives Architecture A its generalisation
story: a new object is a new text query, not a new dataset.

Torch dependency: `detector.py` and `segmenter.py` only. Everything else runs in the FYP venv.
"""
