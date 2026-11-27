# Generating synthetic data with SCMGen

`whydra.scmgen` builds a structural causal model (SCM), generates a sample
from it, and computes ground-truth effects separately. This lets you
evaluate individual treatment effect (ITE), conditional average treatment
effect (CATE), and uplift models on data with a known treatment assignment
and outcome mechanism.

## Quick start

```python
from whydra.scmgen import generate, scenarios

cfg = scenarios.load("rct-noise-y-seg")
ds = generate(cfg, n=1000, seed=42)

observed = ds.to_frame()
print(observed.head())
print("Average structural effect", ds.ate)
```

The `rct-noise-y-seg` scenario simulates randomized treatment and adds
outcome measurement noise in some segments. `generate` uses `seed` to
build the model, generate observations, and apply the changes defined by
the scenario. `scenarios.list_scenarios()` returns the bundled scenarios.

```python
names = scenarios.list_scenarios()
print(names)
```

Use `ScenarioConfig.from_yaml(path)` to load your own scenario from YAML.

## Dataset structure

`ds.to_frame()` returns the observed features, treatment column `t`,
and outcome column `y`. If the scenario distinguishes assignment from
the treatment actually received, the table also includes `z`. Set
`onehot=True` to use a separate indicator column for each level of a
categorical variable with at least three levels.

| Field | Meaning |
| --- | --- |
| `x`, `t`, `y` | Observed covariates, treatment, and outcome. Measurement changes can distort these values. |
| `z`, `t_true` | Treatment assignment and actual treatment, if the scenario distinguishes them. |
| `y0`, `y1` | Potential outcomes for the same row under treatments 0 and 1. |
| `ite` | Realized difference `y1 - y0`. |
| `tau` | Structural effect on the outcome scale with the latent state held fixed. |
| `tau_cate`, `cvte` | Conditional mean and conditional variance of the structural effect. |
| `ate` | Mean of `tau` over the exported rows. |
| `segments`, `graph` | Ground-truth segments and graph with node roles. |
| `pathology_log` | Log of applied changes. |

Train on fields that will be available when you use the model. Use
potential outcomes and structural effects for evaluation.
`truth_targets(ds)` gives the conditions, population, and scale for each
ground-truth field. Check these details when selecting rows or evaluating
a distribution shift. Measurement changes may affect only the observed
data. Changes to the data-generating mechanism may also affect the target
effect.

## Custom scenario

```python
from whydra.scmgen import (
    GraphConfig,
    OutcomeConfig,
    PathologyConfig,
    ScenarioConfig,
    TreatmentConfig,
)

observational = ScenarioConfig(
    name="observational-noise",
    graph=GraphConfig(
        n_confounders=5,
        n_modifiers=3,
        n_predictors=5,
        n_noise=2,
        xx_density=0.1,
        xx_structure="scale_free",
    ),
    treatment=TreatmentConfig(mode="obs"),
    outcome=OutcomeConfig(family="gaussian"),
    pathologies=[
        PathologyConfig(
            kind="noise_y",
            strength=0.8,
            target="segments",
            segment_share=0.25,
        ),
    ],
)
observational.validate()
```

`TreatmentConfig(mode="rct")` assigns treatment at random. With
`mode="obs"`, the probability depends on confounders. The supported
outcome families are `gaussian`, `binary`, `ziln`, `beta`, and `lognormal`.
You can validate the configuration before generating a sample, especially
when combining graph parameters, mediators, and changes to the
data-generating mechanism.

## Clean and modified samples

```python
from whydra.scmgen import generate_pair

clean, dirty = generate_pair(cfg, n=1000, seed=42)
```

`generate_pair` creates one sample and applies the scenario's changes to
a copy. Some scenarios change which rows are present. Check that the
populations align before evaluating individual observations.

Use `generate_split` for a separate test sample from the same model.

```python
from whydra.scmgen import generate_split

clean_train, dirty_train, test = generate_split(cfg, n_train=1000, n_test=300, seed=42)
```

The test sample is generated independently of the training sample.
`test_seed` defaults to `seed + 1`. With `test_dgp="none"`, the test sample
follows the original data-generating mechanism. With `test_dgp="replay"`,
the changes to this mechanism apply to the test sample. Check
`test_target_match` when comparing CATE in the original and modified
populations.

## Reusing a causal model

To reuse the same model with different random streams, build the SCM
once and set separate random seeds.

```python
from whydra.scmgen import apply_pathologies, sample_scm

scm = sample_scm(cfg, model_seed=42)
train = scm.sample(n=1000, sample_seed=7)
holdout = scm.sample(n=300, sample_seed=8)
dirty_train = apply_pathologies(train, seed=43)
```

Here, the samples `train` and `holdout` come from the same graph,
coefficients, and calibration. `apply_pathologies` returns a modified
copy of `train`.

## Saving and loading

Install Parquet support from the repository root.

```bash
python -m pip install -e ".[io]"
```

```python
from whydra.scmgen import load_dataset, save_dataset

path = save_dataset(ds, "out/scmgen-study")
observed, ground_truth, meta = load_dataset(path, verify="hashes")
```

The export writes observations, ground truth, and metadata to separate
files. If the sample has a saved SCM, the export includes its description.
The manifest contains checksums. `load_dataset` returns two tables and a
metadata dictionary. `verify="rows"` checks table sizes against the
manifest. `verify="hashes"` also checks SHA-256 hashes.

If the directory already exists, `overwrite=True` publishes a new version
and updates the `CURRENT` pointer. Each saved dataset has its own directory.
Use `replay` to recreate the sample from the saved SCM and recipe.

```python
from whydra.scmgen import replay

restored = replay(path)
```

Record the Whydra and dependency versions when comparing numerical results.

## Data diagnostics

```python
from whydra.scmgen import data_properties, outcome_profile, truth_targets

correlations = data_properties(ds.x)
profile = outcome_profile(ds)
targets = truth_targets(ds)
```

`data_properties` describes relationships among covariates.
`outcome_profile` shows the outcome distribution and differences between
groups. `truth_targets` helps you pick the matching ground-truth field
for evaluation. For example, use `tau_cate` to evaluate prediction error
for the conditional mean. The difference `y1 - y0` includes individual
variation in the realized outcome.
