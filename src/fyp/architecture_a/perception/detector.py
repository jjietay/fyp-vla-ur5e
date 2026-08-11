""" detector.py

Open Vocabulary Object Detection Model (OWLv2) inference on a single image.
Detections are returned as: (query, score, [x0, y0, x1, y1])

It is sorted by descending score with no thresholding applied, meaning its
completely raw.

Model note: OWLv2 (`google/owlv2-base-patch16-ensemble`) at threshold 0.3 is
what works on this scene. OWL-ViT base was too weak.
"""
from __future__ import annotations


def detect(pil_img, queries, model_name, threshold=None):
    """
    This function takes a photo and a list of text descriptions, and the model
    will output the bounding boxes that matches our description.

    We are supposed to input the model name, in this case we are using OWLv2.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(model_name) # converts to tensor
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval() # switch to evaluation mode

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
