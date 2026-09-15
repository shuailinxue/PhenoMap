# PhenoMap

PhenoMap is a cross-modal framework for mapping patient-level clinical phenotype associations to molecularly interpretable tissue regions by integrating clinically annotated bulk cohorts, spatial transcriptomics, and H&E images. It further enables phenotype-associated regions to be queried in external cohorts using H&E images alone.

![PhenoMap overview](Overview.png)

## Installation

```bash
git clone https://github.com/shuailinxue/PhenoMap.git
cd PhenoMap
conda create -n phenomap python=3.11
conda activate phenomap
pip install -r requirements.txt
```

Optional components use `requirements-optional.txt`; Tutorial 02 reference-label
regeneration uses `requirements-annotation.txt`. Set `HF_TOKEN` when gated
Hugging Face assets require authentication.

## Quick start

Set the data and output paths in the stage YAML files, then run the supported
entry points as needed:

```bash
python step1/scripts/extract_features.py --config step1/configs/default.yaml
python step1/scripts/train.py --config step1/configs/default.yaml
python step1/scripts/predict.py --config step1/configs/default.yaml
python step2/scripts/train.py --config step2/configs/template.yaml
python step3/train.py --mode train --config step3/configs/template.yaml
```

## Documentation

For the complete analysis workflow and tutorials, see the [documentation](https://phenomap.readthedocs.io/en/latest/).


## Contact details

If you have any questions, please contact xueshuailin@whu.edu.cn.

## License

This project is licensed under the MIT License.
