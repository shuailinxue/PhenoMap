# PhenoMap

PhenoMap maps phenotype-associated tissue states by connecting H&E morphology,
spatially resolved cell expression, and phenotype-labeled bulk cohorts.

![PhenoMap overview](Overview.png)

## Workflow

| Stage | Purpose | Standard entry points |
|---|---|---|
| Step1 | Reconstruct cell-level expression from H&E and spot expression | `extract_features.py`, `train.py`, `predict.py` |
| Step2 | Learn cell-level spatial phenotype scores | `train.py`, `infer_tiled.py` |
| Step3 | Query phenotype-associated regions in H&E-only slides | `train.py --mode train/inference` |

## Installation

```bash
git clone https://github.com/shuailinxue/PhenoMap.git
cd PhenoMap
conda create -n phenomap python=3.11
conda activate phenomap
pip install -r requirements.txt
```

Install `requirements-optional.txt` when using Cellpose preprocessing or
optional tutorial visualizations. GPU execution is recommended for feature
extraction and large spatial graphs. Set `HF_TOKEN` in the environment when a
gated Hugging Face model requires authentication.

## Quick start

```bash
python step1/scripts/extract_features.py --config step1/configs/default.yaml
python step1/scripts/train.py --config step1/configs/default.yaml
python step1/scripts/predict.py --config step1/configs/default.yaml

python step2/scripts/train.py --config step2/configs/template.yaml --dry-run
python step2/scripts/infer_tiled.py --help

python step3/train.py --help
```

Copy each YAML template and fill its input paths before running a full workflow.
Detailed stage-specific notes are provided in each step directory.

## Repository layout

```text
step1/          cell-level expression reconstruction
step2/          spatial phenotype learning
step3/          cross-slide phenotype query
preprocessing/  reusable input preparation
baselines/      upstream baseline references
docs/           ReadTheDocs sources
```

Detailed workflows are maintained in the ReadTheDocs sources under `docs/`.

## Contact

Questions and feedback are welcome through GitHub Issues.
