# Step1: Cell-level expression reconstruction

Step1 extracts local and contextual UNI features for segmented cells, builds
spatial cell graphs, and learns cell-level expression under spot-level weak
supervision.

```bash
python step1/scripts/extract_features.py --config step1/configs/default.yaml
python step1/scripts/train.py --config step1/configs/default.yaml
python step1/scripts/predict.py --config step1/configs/default.yaml
```

Copy `configs/default.yaml`, fill the input paths, and provide gated-model
credentials through the `HF_TOKEN` environment variable. Training writes the
expression scaler, gene order, model weights, and optional loss curve to the
configured output directory.
