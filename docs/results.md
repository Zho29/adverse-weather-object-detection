# Recorded results and limitations

The tables below summarize existing JSON artifacts. No full-dataset training or benchmark evaluation was performed during the repository cleanup. Synthetic regression tests and a small CPU training check validate code behavior, not benchmark performance.

## Metric sources

| Condition | Source | Notes |
| --- | --- | --- |
| Clear images | [CLEAR_IMAGE_mAP_BASELINE.json](../experiments/CLEAR_IMAGE_mAP_BASELINE.json) | Nonzero baseline evaluation metrics |
| Foggy images | [foggy_image_map_results.json](../experiments/foggy_image_map_results.json) | Recorded high-fog evaluation metrics |
| Earlier clear-image attempt | [clear_image_map_results.json](../experiments/clear_image_map_results.json) | All-zero metrics; retained for provenance and excluded from the main comparison |

## Clear images

| Model | mAP@50 | mAP@50–95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| COCO-pretrained YOLOv8n | 25.1% | 11.1% | 49.7% | 21.0% |
| YOLO trained on foggy images | 57.6% | 36.1% | 79.1% | 49.4% |
| YOLO trained on WUNET-restored images | 66.0% | 42.2% | 80.3% | 57.2% |

## Foggy images, β = 0.02

| Model | mAP@50 | mAP@50–95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| COCO-pretrained YOLOv8n | 19.3% | 8.8% | 44.6% | 15.6% |
| YOLO trained on foggy images | 65.1% | 41.7% | 79.8% | 56.5% |
| YOLO trained on WUNET-restored images | 53.2% | 34.0% | 77.7% | 43.6% |

Here, “trained on WUNET-restored images” identifies the detector's training input. It does not establish that restoration was applied during every evaluation. See the original evaluation scripts when reconstructing a run.

## Interpretation limits

- **Class definitions differ.** The baseline YAML uses `person/rider`; the data converter uses `person/vehicle`. The pretrained foggy metric description says `person/rider`, and the original COCO class mapping must also be checked. Do not assume these runs share the same label semantics.
- **Reports disagree.** The [original comparison](archive/foggy-detection-comparison.md) contains different headline and table values. It remains an archived report rather than the source of the README table.
- **Run provenance is incomplete.** The JSON files do not record all split definitions, preprocessing, package versions, seeds, or checkpoint hashes. Their comparability has not been independently established.
- **DSNet is excluded from the headline table.** Separate DSNet metric files exist, but a matched evaluation protocol has not been verified.
- **mAP is a detection metric.** These percentages should not be described as classification accuracy or evidence of deployment readiness.

For a reproducible comparison, record the exact dataset split, class mapping, checkpoint, restoration stage, image size, confidence/IoU settings, package versions, hardware, and evaluation command for each run.

## Archived reports

- [Baseline evaluation workflow](archive/baseline-evaluation-readme.md)
- [Baseline mAP report](archive/baseline-results-map.md)
- [Clear-image baseline report](archive/clear-image-baseline-results.md)
- [Final baseline report](archive/final-baseline-results.md)
- [Foggy detection comparison](archive/foggy-detection-comparison.md)

These reports preserve historical observations and may contain server-specific paths or claims that need further validation.

## Corrections affecting future runs

The repaired restoration pipelines consistently normalize RGB inputs to `[-1, 1]` and convert tanh outputs back to display pixels. PSNR now uses the correct data range of 2 for these normalized tensors and averages per-image measurements. Using range 2 instead of 1 adds approximately 6.02 dB for the same image error; older PSNR values are not directly comparable. Saved historical results were not rewritten.

The DSNet custom head now decodes all three prediction slots per scale, uses actual grid dimensions, applies class-aware NMS, and maps classes to COCO IDs correctly. Its training loss now supervises box coordinates and classes, which the old loss omitted. Historical checkpoints therefore need retraining for meaningful detection quality. New checkpoints record `detection_box_format="sigmoid"`; older checkpoints use the explicit legacy decoder.

Top-level legacy evaluation scripts, including `evaluate_clear_map.py`, retain original server paths and can compare incompatible pretrained/custom class mappings. Use matched class definitions and the repaired CLIs when producing new results. These repairs do not validate the historical headline metrics.
