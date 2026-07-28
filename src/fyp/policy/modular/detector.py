"""Open-vocabulary object detection. Architecture A stage 2.

Model inference ONLY — the filtering logic lives in `filters.py` so it stays
free of torch. This is the only module in `policy/modular/` that needs the
lerobot venv.

Returns detections as `(query, score, [x0, y0, x1, y1])` triples in pixel
coordinates, sorted by descending score, with NO thresholding applied: the
caller filters, so the raw scores stay inspectable.

Model note: OWLv2 (`google/owlv2-base-patch16-ensemble`) at threshold 0.3 is
what works on this scene. OWL-ViT base was too weak.
"""
from __future__ import annotations


def detect(pil_img, queries, model_name, threshold=None):
    import torch


    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    inputs = processor(text=[queries], images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)


    target_sizes = torch.tensor([pil_img.size[::-1]], device=device)


    post = getattr(processor, "post_process_grounded_object_detection", None)\
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
