"""segmenter.py

It takes a photo and the detector's boxes and gives you one pixel mask per box,
True on the object and False everywhere else, from Segment Anything (SAM).

Why (13 Oct 2026). Picks closed BEHIND the orange by about half its width, and
more so the further the orange was from the camera. The box-centre depth patch
sits on a small round object whose edges the stereo matcher blends with the
table behind it, so it read deep, and the centre push then added a radius on
top. A box is mostly not the object near its edges; a mask is only the object.
This is the step published open-vocabulary pickers add between the detector and
the 3D position (OK-Robot: OWL-ViT boxes refined into SAM masks).

The box is SAM's prompt, so SAM never decides WHAT to segment, only its outline.
Architecture A's vocabulary still comes from grounding and OWLv2 alone.

Torch, so lerobot venv only, like detector.py. The model is loaded once per
process and kept, for the same latency reason as the detector.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=2)
def _load(model_name: str):
    """It takes a SAM checkpoint name and gives you (processor, model, device), loaded once."""
    try:
        import torch
        from transformers import SamModel, SamProcessor
    except ImportError as e:
        raise ImportError(
            f"SAM needs torch and transformers, which live in the lerobot venv ({e}). "
            "From the FYP folder run the script as:\n"
            "    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/<script>.py") from e
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return processor, model.to(device).eval(), device


def segment(pil_img, boxes, model_name: str) -> list[tuple[np.ndarray, float]]:
    """
    It takes a photo and pixel boxes (x0, y0, x1, y1) and gives you, per box, a
    boolean mask the size of the photo and SAM's own quality score for it.

    One mask per box (multimask_output=False): the box already says which
    object, so SAM's three-way "part, object, whole" choice is not wanted.
    """
    if not boxes:
        return []
    processor, model, device = _load(model_name)
    import torch

    inputs = processor(pil_img, input_boxes=[[list(map(float, b)) for b in boxes]],
                       return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    masks = processor.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())[0]          # (n_boxes, 1, H, W)
    scores = out.iou_scores.cpu().numpy().reshape(len(boxes), -1)[:, 0]
    return [(masks[i, 0].numpy().astype(bool), float(scores[i])) for i in range(len(boxes))]
