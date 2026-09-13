import numpy as np
import pandas as pd

from step2.utils.clinical_analysis import (
    benjamini_hochberg,
    define_pr_groups,
    fit_repeated_l1_cox,
)


def test_benjamini_hochberg_is_monotone_in_rank_order():
    raw = np.array([0.04, 0.001, 0.03, 0.20])
    adjusted = benjamini_hochberg(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all((adjusted >= raw) & (adjusted <= 1))


def test_define_pr_groups_selects_exact_top_fraction():
    cells = pd.DataFrame(
        {
            "beta": np.arange(20, dtype=float),
            "cell_type": ["Cancer_epithelial"] * 20,
        },
        index=[f"cell-{index}" for index in range(20)],
    )
    grouped, threshold = define_pr_groups(cells, high_fraction=0.05)
    assert grouped["pr_group"].eq("PR-high (top 5%)").sum() == 1
    assert grouped.loc["cell-19", "pr_group"] == "PR-high (top 5%)"
    assert threshold == 19


def test_repeated_l1_cox_uses_strongest_eligible_penalty_and_positive_coefficients():
    rng = np.random.default_rng(7)
    patients = [f"patient-{index}" for index in range(80)]
    genes = [f"gene-{index}" for index in range(8)]
    values = rng.normal(size=(len(genes), len(patients)))
    risk = 0.8 * values[0] - 0.4 * values[1]
    event_time = rng.exponential(scale=np.exp(-risk) * 20)
    censor_time = rng.exponential(scale=30, size=len(patients))
    clinical = pd.DataFrame(
        {
            "OS.time": np.minimum(event_time, censor_time) + 0.01,
            "OS.event": (event_time <= censor_time).astype(int),
        },
        index=patients,
    )
    expression = pd.DataFrame(values, index=genes, columns=patients)

    signature, diagnostics = fit_repeated_l1_cox(expression, clinical, genes)
    best = diagnostics.loc[diagnostics["best_mean"]].iloc[0]
    selected = diagnostics.loc[diagnostics["one_se"]].iloc[0]

    assert selected["alpha"] >= best["alpha"]
    assert selected["mean_cindex"] >= best["mean_cindex"] - best["se"]
    assert signature["coefficient"].gt(0).all()
