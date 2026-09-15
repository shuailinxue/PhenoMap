# Baselines

PhenoMap does not vendor complete third-party repositories. Reproduce comparisons
against the upstream projects and record the exact release or commit used.

## Step 1 expression-generation baselines

- [GHIST](https://github.com/SydneyBioX/GHIST)
- [iStar](https://github.com/daviddaiweizhang/istar)
- [sCellST](https://github.com/loicchadoutaud/sCellST)

## Step 3 phenotype-query baselines

- [KEEP](https://github.com/biomed-AI/KEEP)
- PathPT, implemented as prompt tuning of a frozen KEEP encoder using source-side
  support annotations

The `step3/baselines` package contains lightweight reproducibility wrappers for
KEEP and PathPT inference, not vendored copies of their upstream code.
Baseline-specific environments should remain separate from the PhenoMap environment.
