# Step2: Spatial phenotype mapping

Step2 aligns spatial cell expression with phenotype-labeled bulk cohorts. The
`PhenoMapPhenotypeModel` combines a shared expression encoder, graph attention,
optimal-transport alignment, and Cox or binary phenotype supervision.

```bash
python step2/scripts/train.py --config step2/configs/template.yaml --dry-run
python step2/scripts/train.py --config step2/configs/template.yaml
python step2/scripts/infer_tiled.py --help
```

The template supports ROI or spatial-block training, optional cell filters, and
an optional auditable global orientation of cell-level phenotype scores.
