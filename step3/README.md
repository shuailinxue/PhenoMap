# Step3: Cross-slide phenotype query

Step3 trains PhenoMap adapters over frozen KEEP vision/text and scGPT expression
encoders, then transfers a Step2 phenotype signal to H&E-only slides.

```bash
python step3/train.py --mode train --config step3/configs/template.yaml
python step3/train.py --mode inference --config step3/configs/template.yaml \
  --ckpt_path checkpoints/step3/last.ckpt
```

Copy the template and fill the source data paths before training. Inference
reads images from `data.he_folder` and writes maps to `data.output_dir`.
