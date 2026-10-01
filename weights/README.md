# Pretrained weights

Weights-only checkpoints (no optimizer state). Each `.pt` is a dict with
`model` (state_dict), `model_cfg` (`CrackViTConfig` kwargs), `epoch` and `val_metrics`.

| dataset | file | model | split | best epoch | val AUC | val F1 | val MCC |
|---|---|---|---|---|---|---|---|
| SDNET2018 | `sdnet2018/loopcrackvit_gipa_full_sdnet2018.pt` | GIPA_FULL, conv stem, core 6 x 2 passes, 3.9 M params | random (seed 42), SMOTE-style balancing | 46 | 0.937 | 0.753 | 0.710 |
| METU | `metu/` | — | — | — | — | — | — |

SHA-256 `sdnet2018/loopcrackvit_gipa_full_sdnet2018.pt`:
`3d7960b8670a96b4cc970ef856cf7dcd2155440157eaca6fea75e59e1ea946be`

Training command (SDNET2018):

```
python train.py --split-mode random --scheduler plateau --weight-decay 0 --grad-clip 0 \
  --early-stop-min-delta 0 --early-stop-patience 10 --monitor val_auc --lr 0.0003 \
  --batch-size 16 --epochs 60 --imbalance smote --save-every 1
```

## Loading

```python
import torch
from model import CrackViTConfig, LoopedCrackViT

ck = torch.load("weights/sdnet2018/loopcrackvit_gipa_full_sdnet2018.pt", map_location="cpu", weights_only=True)
model = LoopedCrackViT(CrackViTConfig(**ck["model_cfg"])).eval()
model.load_state_dict(ck["model"])
# inputs: RGB, resized to 224x224, normalised with mean = std = (0.5, 0.5, 0.5)
p_crack = torch.sigmoid(model(x)[-1])      # model(x) returns one logit per exit: [exit1, final]
```
