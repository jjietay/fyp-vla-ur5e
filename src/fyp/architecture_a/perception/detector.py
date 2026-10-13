""" detector.py

Open Vocabulary Object Detection Model (OWLv2) inference on a single image.
Detections are returned as: (query, score, [x0, y0, x1, y1])

It is sorted by descending score with no thresholding applied, meaning its
completely raw.

Model note: OWLv2 (`google/owlv2-base-patch16-ensemble`) at threshold 0.3 is
what works on this scene. OWL-ViT base was too weak.

QUERY FORMAT IS NOT A STYLE CHOICE. Each query must be a short noun phrase
("red cube"), never a sentence ("place that red cube into that metal tray").

The reason is architectural rather than a length limit. OWLv2 encodes text with
CLIP's text encoder, and every query becomes ONE embedding that is matched
against the per-patch embeddings of the image. A sentence therefore produces a
single embedding for the whole sentence, and no region of the image corresponds
to it, so the match is meaningless. There is no mechanism inside the model for
splitting a sentence into the several objects it mentions, and no mechanism for
ignoring the verbs, articles and destinations it also contains. CLIP's 77-token
context is a real limit but it is not the binding one here: a sentence fits
comfortably and still fails.

This is precisely why Architecture A needs a language model in front of the
detector. Something has to turn "place that red cube into that metal tray" into
["red cube", "metal tray"] before OWLv2 can be asked anything at all. See
`architecture_a/planner.py::extract_queries`, which is that step.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _load(model_name):
    """
    It takes a model name and gives you (processor, model, device), loaded from
    disk once per process and reused after that.

    Loading is the slow part, not inference. Until 13 Oct 2026 this ran inside
    `detect()` on every call, and the 1 Oct traces show the cost: 23 detection
    stages, 10.7 s median and none under 6 s, inside A's re-perception loop. That is load time recorded as A's latency in the
    comparison. Keyed by name, so a second model can still be tried in the same
    process.
    """
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError as e:
        # The FYP venv has no torch by design, so this is the usual mistake, and
        # the bare "No module named 'torch'" says nothing about the fix
        raise ImportError(
            f"OWLv2 needs torch and transformers, which live in the lerobot venv, not "
            f"the FYP one ({e}). From the FYP folder run the script as:\n"
            f"    PYTHONPATH=$PWD/src ~/lerobot/.venv/bin/python scripts/<script>.py") from e

    processor = AutoProcessor.from_pretrained(model_name) # converts to tensor
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval() # switch to evaluation mode
    return processor, model, device


def detect(pil_img, queries, model_name):
    """
    This function takes a photo and a list of text descriptions, and the model
    will output the bounding boxes that matches our description.

    We are supposed to input the model name, in this case we are using OWLv2.

    There is no threshold argument on purpose. Everything comes back, sorted by
    descending score, and `filters.apply_filters` decides what to keep. Keeping
    the model call pure is what lets the filtering be unit tested in a venv with
    no torch in it, and it is why the counts at each filter stage can be reported
    in the trace.
    """
    # Loaded once per process, see `_load`. The first call still pays the load.
    # Before `import torch` on purpose, so a missing torch gets `_load`'s message.
    processor, model, device = _load(model_name)
    import torch

    inputs = processor(text=[queries], images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad(): # no need to calculate gradient for backpropagation
        outputs = model(**inputs)


    target_sizes = torch.tensor([pil_img.size[::-1]], device=device)


    post = getattr(processor, "post_process_grounded_object_detection", None) \
        or processor.post_process_object_detection
    try:
        results = post(outputs, threshold=0.0, target_sizes=target_sizes,
                       text_labels=[queries])[0]
    except TypeError:
        results = post(outputs, threshold=0.0, target_sizes=target_sizes)[0]

    text_labels = results.get("text_labels")
    int_labels = results.get("labels")

    dets = []
    for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
        if text_labels is not None and text_labels[i] is not None:
            q = text_labels[i]
        else:
            q = queries[int(int_labels[i])]
        dets.append((q, float(score), [round(float(v), 1) for v in box.tolist()]))
    return sorted(dets, key=lambda d: -d[1])
