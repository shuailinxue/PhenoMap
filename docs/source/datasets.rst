Datasets
========

The repository stores code and configuration files. Raw slides, expression
matrices, masks, checkpoints and large generated outputs should be stored
outside the source tree.

Required file types
-------------------

For a new cohort, prepare the relevant subset of:

* H&E image files: ``.tif``, ``.tiff``, ``.png`` or ``.jpg``.
* Cell/nucleus instance masks where background is ``0`` and each cell has a
  positive integer ID.
* Spot-level expression matrices and spot coordinates for Step 1.
* Cell-level expression matrices and cell coordinates for Step 2.
* Bulk expression matrices and matched clinical tables for phenotype learning.
* Optional cell-type annotations and ROI masks for filtering, target adaptation
  and evaluation.
