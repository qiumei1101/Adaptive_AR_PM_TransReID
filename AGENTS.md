# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a Python deep learning research project for **Vehicle Re-Identification** using Vision Transformers (ViT/DeiT). The core workflow builds 3 TransReID models at different input aspect ratios and adaptively fuses their features at inference time. There is no web UI, no database, and no multi-service architecture.

### Missing modules

The original repository was incomplete — several modules referenced by `model/make_model.py` were absent:

- `loss/metric_learning.py` (Arcface, Cosface, AMSoftmax, CircleLoss) — stub implementations were added.
- `model/backbones/resnet.py` (ResNet, Bottleneck) — stub added. Only used when `MODEL.NAME != 'transformer'`; all configs use `'transformer'`.
- `model/__init__.py` and `model/backbones/__init__.py` — created for proper package imports.
- Name aliases (`vit_base_patch16_224_TransReID` etc.) were appended to `adptive_ar_PM_TransReID.py` because `make_model.py`'s factory dict references names without the `_PM_` suffix while only `_PM_` variants were defined.

### Running the code

- **No GPU required for basic verification.** Models can build and run forward passes on CPU by overriding `MODEL.PRETRAIN_CHOICE` to `'self'` (skips loading pretrained ImageNet weights). Use `--device cpu`.
- **Full inference** requires: (1) trained model weight files (`.pth`), (2) ReID dataset images (VeRi-776, VehicleID, etc.), and (3) ImageNet-pretrained ViT checkpoint (`jx_vit_base_p16_224-80ecf9dd.pth`). See `README.md` for CLI usage.
- Entry points: `dynamically_ar_fusing_veri.py` (VeRi-776) and `dynamically_ar_fusing_veh.py` (VehicleID).

### Lint

```bash
python3 -m flake8 --select=E9,F63,F7,F82 *.py model/ config/ utils/ datasets/ loss/
```

### Key caveats

- No automated tests exist in this repository.
- No CI/CD pipeline exists.
- Dataset paths are hardcoded in some files (e.g., `datasets/vehicleid.py`).
- `dynamically_ar_fusing_veh.py` imports and instantiates `VehicleID()` at module level (line 21-22), which will fail if the VehicleID dataset is not at the expected path. Only `dynamically_ar_fusing_veri.py` can run cleanly without datasets present.
