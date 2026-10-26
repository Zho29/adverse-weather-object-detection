# Contributing

Describe the problem, the proposed change, and how you verified it. Keep model or preprocessing changes separate from formatting changes so results remain interpretable.

## Development

Create a virtual environment and install [requirements.txt](requirements.txt). See [setup](docs/setup.md) for datasets and optional DINO dependencies.

Before opening a pull request:

```bash
python -m compileall -q experiments video_detection_analysis tests
python -m unittest discover -s tests -v
bash -n experiments/create_yolo_dataset_beta0.02.sh
bash -n experiments/fix_labels_beta0.02.sh
bash -n experiments/install_mmdetection.sh
```

Run a relevant model or dataset check when changing behavior. Some `test_*.py` files are historical evaluation scripts with top-level execution and server-specific paths; they are not an isolated unit-test suite.

## Experiment changes

Include the dataset split, class-ID mapping, preprocessing, training/evaluation command, package versions, seed, checkpoint identity, hardware, and metrics. Link raw results and distinguish reproduced results from archived observations.

Keep datasets, virtual environments, credentials, and temporary output out of Git. Use `outputs/` for new scratch runs and Git LFS for intentionally shared model files. Do not overwrite historical metrics or checkpoints without explaining their provenance.

## Issues and pull requests

For a bug, include the command, traceback, environment, and expected behavior. For a result discrepancy, identify both artifacts and their evaluation protocols. Check the [known limitations](docs/results.md) before interpreting comparisons.
