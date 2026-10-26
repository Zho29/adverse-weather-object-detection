# WeatherUNet

A PyTorch U-Net restoration model and paired foggy/clear image training workflow. This implementation is based on the research paper stored in [2407.02581v1.pdf](2407.02581v1.pdf). Paper equivalence and reported paper metrics have not been independently reproduced here.

## Files

| File | Purpose |
| --- | --- |
| `weather_unet.py` | Encoder–decoder with skip connections and RGB restoration output |
| `foggy_dataset.py` | Foggy Cityscapes paired-image loader |
| `train_wunet_foggy.py` | Dataset CLI, training, validation, and checkpoint saving |
| `train_wunet.py` | Earlier training experiment |

## Train

Install the root [requirements](../../requirements.txt), then run from the repository root:

```bash
python experiments/WUNET/train_wunet_foggy.py \
  --root-dir /path/to/Foggy_Cityscapes_dataset \
  --beta 0.02 --epochs 200 \
  --save-dir experiments/checkpoints/wunet_foggy
```

Supported fog densities are `0.005`, `0.01`, and `0.02`. See [setup](../../docs/setup.md) for the dataset layout and checkpoint handling.

## Model usage

When running Python from this directory:

```python
import torch
from weather_unet import WeatherUNet

model = WeatherUNet(n_channels=3, n_classes=3, bilinear=True).eval()
with torch.no_grad():
    restored = model(torch.randn(1, 3, 256, 256))
```

This example uses random weights. Load a compatible checkpoint for restoration experiments. The foggy dataset loader normalizes RGB inputs to `[-1, 1]`, matching the model's tanh output.
