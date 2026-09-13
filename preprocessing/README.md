# Preprocessing

This directory contains only reusable input preparation commands:

- `segment_cells.py` creates an instance mask from a large H&E image with tiled
  Cellpose inference.
- `build_pseudo_visium.py` aggregates cell expression into synthetic spot-level
  supervision for Step1.

Dataset- and cohort-specific conversion scripts are intentionally excluded.
