# Supervised tree / random-forest transfer — readable notebook

## Open first

`notebooks/Tree_Forest_Transfer_Classification.ipynb`

This is a **classification-only** reorganization of the original draft. The source distribution is the accepted seven-cluster Gaussian law in `source_distribution.json` (no stored source dataset is needed).

**Choice of model** — first code cell:

```python
MODEL_KIND = 'forest'  # or 'tree'
TREE_PARAMS = dict(max_depth=6, min_samples_leaf=5)
FOREST_PARAMS = dict(n_estimators=10, max_depth=3, min_samples_leaf=1)
```

## How to edit a scenario

Each of six numbered sections has **three code blocks**:

1. **Settings:** edit `SCENARIO_PARAMS`, `STRENGTH`, `SOURCE_N`, `TARGET_N`, `SOURCE_TEST_N`, `TARGET_TEST_N`, `SEED`. This block documents every changed cluster parameter — nothing is hidden in a global random recipe.
2. **Data + baselines:** new source/target samples, Bayes contours and ellipses; explicit `.fit()` for source-only, target-only and unadapted pooling; predictions on both domains.
3. **Native ADAPT:** explicit `TransferTreeClassifier` or `TransferForestClassifier`, `.fit(Xt_train, yt_train)` for relabeling, SER and STRUT; optional `TransferForestSelector.model_selection(...)` and forest strategy bars; held-out scores and figures.

The scenarios are in this order:

- 1. Same distribution, new independent draw.
- 2. Strict feature-only covariate selection, no change to P(Y | X).
- 3. **Translations only**; each cluster displacement is editable.
- 4. Remove one cluster and add another, with explicit new center, axes, orientation, weight and class.
- 5. Relabel chosen clusters (default `[2, 4]`, matching the previously fixed selection from seed 131).
- 6. Small mixed transformations (one translation, rotation, axis stretch and weight change, all explicitly listed).

At the bottom, §7 lets you select a saved scenario by name and edit target-size budgets, shift-strength grid and repetition count. Set `RUN_SWEEPS=True` to run the curves. Because addition/removal and relabeling are categorical, they do **not** have a misleading continuous shift-severity sweep.

## Tree visualization conventions

When `MODEL_KIND='tree'`, the transferred tree and source tree can be plotted side-by-side (first `TREE_PLOT_DEPTH` levels). The colors are:

- Blue node backgrounds: majority class 0; orange node backgrounds: majority class 1.
- **Solid red border**: pruning, expansion, or changed split feature.
- **Dashed red border**: changed threshold while retaining the split feature.
- Amber border: leaf majority class changed without restructuring.

Node comparison uses left/right decision paths up to the first structural mismatch; beyond that point a valid one-to-one node correspondence generally does not exist. Class background is based on fitted `tree_.value` at each node.

When `MODEL_KIND='forest'`, tree drawings are skipped. Selector bars show the **fraction of trees** assigned to `src`, `trgt`, `relab`, `ser`, `strut`, *not* accuracy scores.

## Execution

Inside a compatible existing conda environment:

```bash
conda install numpy scipy pandas matplotlib scikit-learn nbformat nbclient ipykernel jupyterlab
python -m pip install adapt
jupyter lab
```

Open the notebook from either the repository root or its `notebooks/` directory. It automatically adds the local `transfer_tree_demo` package to Python's import path.

The notebook was fully executed in **both model modes** for generation, source-only, target-only and pooled baselines, with no fitting errors. Since **ADAPT is not available** in this execution environment, SER/STRUT/selector results are *not* present in the precomputed notebook. Their actual ADAPT calls remain visible and will execute in a compatible ADAPT environment.

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

11 tests pass, including strict-covariate behavior, deterministic cluster labels, reproducibility and the dashed/solid tree-border logic.

## Project files

- `source_distribution.json`: fixed seven-component source probability law.
- `transfer_tree_demo/distribution.py`: JSON loader and sampler.
- `transfer_tree_demo/scenarios.py`: directly parameterized, reusable target law builders.
- `transfer_tree_demo/plotting.py`: figures and tree annotations; **no model fitting**.
- `notebooks/Tree_Forest_Transfer_Classification.ipynb`: visible fitting and evaluation commands.
- `results_refactored/forest/`: genuine baseline figures already embedded in the notebook.
- `results/tree_colors_demo.png`: style diagnostic from two ordinary sklearn trees, **not an ADAPT transfer result**.

## Repeated sampling from fixed distributions (new)

Each scenario now contains an additional **D. Repeated-data evaluation** block immediately after the original single-run visualization. Set, **inside that scenario's settings cell**:

```python
RUN_REPETITIONS = True
N_REPETITIONS = 8  # e.g. 20-30 for a more robust comparison
```

Each repetition draws new *source train, target train, source test, and target test*
observations using new seeds, but keeps `source_distribution.json`, the selected
scenario recipe, its `STRENGTH`, and the sample sizes **fixed**. It fits fresh
source-only, target-only, pooled, relabel, SER and STRUT predictors (and a forest
selector in forest mode) and evaluates them on the **same test samples within
that repetition**. ADAPT methods are only included if genuine ADAPT fits succeed.

For each scenario you get:

- `<scenario>_repeated_scores.csv`: one row per repetition and method.
- `<scenario>_04_repeated_accuracy.png`: individual scores plus mean ± SD.
- `<scenario>_paired_gains.csv`: transferred-method paired differences against
  the stronger of source-only/target-only and against pooled training, when available.
- `<scenario>_05_repeated_selector.png`: `src`/`trgt`/`relab`/`ser`/`strut`
  mean selection fractions across successful forest selections, when available.
- `all_scenarios_repeated_summary.csv`: mean, SD and sample counts, grouped by
  scenario and method.

The **single-run** source/target and prediction figures are still separate,
so the examples remain easy to inspect. The repeated-sampling experiments and
optional **§7 target-budget/shift-strength sweeps** are independent; you can
run one without the other.

No transfer score is fabricated if the native ADAPT package is unavailable.
