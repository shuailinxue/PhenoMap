Datasets and reproducibility contract
=====================================

PhenoMap keeps source code and small configurations in Git. Raw slides,
expression matrices, masks, checkpoints and generated arrays remain external.
Tutorials 01–06 resolve these resources through ``phenomap.data`` rather than
assuming a directory hierarchy or a machine-specific filename.

Configure tutorial data
-----------------------

Copy the public example and edit only the values:

.. code-block:: bash

   cp configs/tutorial_data.example.yaml configs/tutorial_data.local.yaml

``configs/tutorial_data.local.yaml`` is ignored by Git. When several manifests
are needed, select one explicitly:

.. code-block:: bash

   export PHENOMAP_TUTORIAL_CONFIG=/path/to/tutorial_data.yaml

The loader checks the environment variable first, then the ignored local file,
then ``configs/tutorial_data.example.yaml``. Relative values are resolved from
the repository root. Missing resources raise an error naming the semantic key
that must be configured; resolved private paths are not printed.

Canonical resource roles
------------------------

``hbc1``
   ``he_image`` and ``nucleus_mask`` are the registered H&E image and integer
   instance mask. ``cell_features`` is the Step 1 HDF5 cache with ``features``,
   ``context_features`` and ``metadata`` keys. ``pseudo_spots``,
   ``spot_locations``, ``spot_diameter`` and ``gene_list`` provide Step 1
   supervision. ``measured_expression`` and ``generated_expression`` are
   cell-by-gene CSV tables with cell IDs in the first column.

   ``reference_marker_table`` and ``reference_atlas`` provide the external
   breast reference resources. ``marker_cache``, ``phenomap_cell_types``,
   ``measured_cell_types``, ``celltypist_probabilities``,
   ``tangram_probabilities``, ``ingest_labels`` and ``reference_cell_types``
   support Tutorial 02. Annotation tables use a cell ID index and a
   ``cell_type`` column in the canonical form; the ensemble result also uses
   ``decision_level``. Probability tables use cell IDs as rows and cell types
   as columns.

   ``phenotype_scores`` is a one-column Parquet table indexed by cell ID; the
   canonical score column is ``phenotype_score``. ``differential_expression``
   contains ``names``, ``scores``, ``logfoldchanges`` and ``pvals_adj``.
   ``spatial_expression`` is an AnnData file with cell coordinates in
   ``obsm['spatial']``. ``step1_checkpoint``, ``step2_config``,
   ``source_checkpoint`` and ``step3_config`` are model or regeneration
   resources. PathPT prompt tuning uses the HBC1 source-side resources
   ``pathpt_support_image`` and ``pathpt_support_mask``; target-cohort images
   and masks must not be used as support.

``hbc2``
   ``he_image``, ``nucleus_mask``, ``measured_expression`` and
   ``reference_cell_types`` describe the independent target section.
   ``phenomap_query_map``, ``keep_query_map`` and ``pathpt_query_map`` are
   aligned 2-D NumPy score arrays. ``phenomap_concordance_map`` is the accepted
   source-only map used for molecular concordance. ``molecular_pr_scores`` is a
   one-column cell-indexed Parquet table and ``cell_coordinates`` is the
   reusable coordinate cache.

``cohorts``
   ``tcga_expression`` and ``tcga_clinical`` are tab-delimited reference-cohort
   tables. The clinical table must contain ``OS.time`` and ``OS.event``.
   ``gse103091_*`` and ``gse7390_*`` are tab-delimited validation tables; their
   clinical tables must contain ``OS.time``, ``OS.event``, ``MFS.time`` and
   ``MFS.event``. Expression matrices use genes as rows and samples as columns.

``cross_center`` and ``her2``
   ``images`` and ``masks``/``proxy_masks`` are directories of matched PNG
   files. Each method-specific score-map directory contains
   ``<image_stem>_score_map.npy``. ``evaluation_cache`` stores the accepted
   slide- or ROI-level metric table.

Cache and regeneration behavior
-------------------------------

Tutorials reuse every valid expensive artifact. Tutorial 01 regenerates
features, a full Step 1 model and expression only when the corresponding cache
is absent. Tutorial 02 regenerates markers, annotation, optional transfer
labels and the ensemble in dependency order. Tutorial 03 runs the configured
Step 2 workflow only when phenotype scores are absent. Tutorial 04 likewise
reuses phenotype scores and differential expression before running lightweight
analysis.

Tutorial 05 can regenerate KEEP and PathPT maps when their optional model
dependencies and source-side support inputs are available. A missing PhenoMap
query map requires both ``hbc1.source_checkpoint`` and the final
paper-compatible ``hbc1.step3_config``; the tutorial never substitutes a toy or
default configuration. Tutorial 06 requires external cohort images and proxy
masks. Missing PhenoMap maps likewise require both canonical Step 3 resources;
optional baseline maps may be supplied as precomputed artifacts. Missing items
are reported by semantic manifest key.

External resources
------------------

The breast single-cell atlas, bulk and validation cohorts, target-cohort
slides, proxy masks, foundation-model weights and optional baseline outputs are
not distributed with this repository. Obtain them under their original data
licenses and point the manifest at the local copies. Set ``HF_TOKEN`` only as
an environment variable when a gated model requires authentication.
