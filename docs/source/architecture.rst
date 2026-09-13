Architecture
============

PhenoMap is organized as a three-stage workflow. Each stage can be run independently
for debugging, ablation and reuse on new cohorts.

Step 1: expression super-resolution
-----------------------------------

Step 1 predicts cell-scale gene expression from histology and spot-level spatial
transcriptomics. The model extracts local and contextual H&E features for each
segmented cell, constructs a radius graph over cell centroids and trains a graph
model under weak spot-level supervision.

Key modules:

* ``step1/scripts/extract_features.py`` extracts UNI-style cell features.
* ``step1/scripts/train.py`` runs the canonical training workflow.
* ``step1/scripts/predict.py`` exports a cell-by-gene prediction matrix.
* ``step1/models/spatial_model.py`` implements the spatial expression model.

Step 2: spatial phenotype learning
----------------------------------

Step 2 links spatial cell expression to bulk clinical phenotypes. A shared
expression encoder maps spatial cells and bulk samples into a common latent
space. Graph attention layers encode local tissue context, while Sinkhorn
alignment relates cell states to bulk samples. Cox and binary phenotype losses
are supported.

Key modules:

* ``step2/scripts/train.py`` trains phenotype models from YAML configs.
* ``step2/scripts/infer_tiled.py`` scores large spatial graphs by tiled
  inference.
* ``step2/models/phenotype_model.py`` implements the graph phenotype model,
  Cox loss and alignment utilities.
* ``step2/utils/score_orientation.py`` provides auditable phenotype-score
  orientation.

Step 3: multimodal phenotype transfer
-------------------------------------

Step 3 transfers Step2 phenotype signals from molecularly profiled source tissue
to H&E-only target slides. Frozen KEEP vision/text encoders and a frozen scGPT
encoder provide foundation representations. PhenoMap trains lightweight adapters to
map H&E patches into gene-aligned and phenotype-aligned spaces.

Key modules:

* ``step3/train.py`` runs training and inference.
* ``step3/models/adapters.py`` implements PhenoMap adapters and phenotype heads.
* ``step3/data/dataset.py`` implements source, target and inference datasets.

Design principles
-----------------

* Keep source-domain molecular supervision and target-domain mask supervision
  separable.
* Store paths, task definitions and hyperparameters in YAML files.
* Save intermediate cell-level outputs so that downstream analysis can be
  repeated without rerunning upstream neural models.
* Treat large generated artifacts as external experiment outputs rather than
  source-code files.
