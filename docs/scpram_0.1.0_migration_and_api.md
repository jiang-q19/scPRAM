# scPRAM 0.1.0 migration and API guide

## What this version changes

scPRAM learns a latent representation, matches source control and perturbed cells
with optimal transport, then transfers source changes to target control cells
using attention. This release makes failures explicit and exposes configuration.
It does not demonstrate better perturbation prediction on biological datasets.

| Area | Historical code | 0.1.0 |
| --- | --- | --- |
| Training | Fixed epochs, final batch loss display | Sample-weighted epoch metrics, optional validation/early stopping |
| Latents at prediction | Random posterior sample | Posterior mean; `sample=True` enables sampling |
| Top-k attention | First target's weights reused | Per-target weights; `attention_mode="legacy_first"` available |
| OT | Fixed EMD iteration limit, unchecked result | Configurable EMD/log-domain Sinkhorn, bounded retries, status and marginal checks |
| Input | Whole matrix via `to_df()` | Sparse data retained; only minibatch rows become dense float32 |
| Cell-type mean path | Random balancing/subsampling | Direct means over paired source types |
| Device | Separately stored device could become stale | Actual parameter device, automatic initial placement |
| Packaging | Old pinned audio/vision/notebook stack and Windows requirements | Source installation, core dependencies and optional extras |

These choices affect scientific comparability. Label variants separately; do not
choose the best variant using held-out target scores. Training architecture,
parameter names, and the weighted reconstruction/KL objective remain compatible
with old state dictionaries. Python >=3.10 and PyTorch >=2.6 are now required.

## Training

Existing positional calls remain valid:

```python
model.train_SCPRAM(train, 100, 128, 5e-4)
```

New keyword arguments: `validation_adata=None`, `validation_fraction=0.0`,
`patience=None`, `min_delta=0.0`, `restore_best=True`, `seed=None`, `layer=None`,
`weight_decay=1e-5`, `grad_clip=10.0`, `verbose=True`.

Use either an explicit validation AnnData or a random split, not both.
`patience` requires validation; no validation is silently synthesized. For
biological replicates or donors, provide an explicit group-disjoint validation
set instead of assuming a random cell split measures generalization. When
`restore_best=True`, the lowest observed validation-loss weights are restored,
even if all epochs run. `min_delta` controls patience resets; tiny improvements
still count when selecting the best checkpoint. Early stopping occurs after
`patience` consecutive epochs without a sufficiently large improvement.

`history_` contains sample-weighted training loss, reconstruction loss, KL loss,
and (when provided) validation loss. Validation runs without input noise and
uses posterior means, so it is stable but is not the stochastic training ELBO
estimate and is not a measure of perturbation accuracy. `best_epoch_` is 1-based;
`stopped_early_` records whether patience terminated the loop. `fit` returns the
model; `train_SCPRAM` returns a copy of the history. Each call starts a new Adam
optimizer and history; this is not optimizer-state training resumption.

Use constructor `seed=...` for initialization and training `seed=...` for the
split, shuffling, noise, and sampling. Seeds do not promise bitwise equality
across devices, BLAS libraries, or PyTorch versions.

## Optimal transport

```python
from scpram import OTConfig, OTConvergenceError

exact = OTConfig(method="emd", max_iter=1_000_000, retries=1)
entropic = OTConfig(method="sinkhorn", max_iter=10_000, reg=0.05,
                    tolerance=1e-7, retries=1)
```

With omitted `max_iter`, EMD uses 1,000,000 and Sinkhorn uses 10,000. A retry
restarts the same objective with an iteration budget multiplied by
`retry_multiplier=5`. No automatic switching of solver or regularization occurs.
EMD requires an optimal solver status in addition to feasible masses. Sinkhorn
requires solver convergence and checked masses. Nonfinite or negative plans are
rejected. `OTConvergenceError.diagnostics` contains attempt budgets, warnings,
and marginal errors. A failed plan is never consumed.

Both solvers use uniform cell masses and Euclidean distances in float64. Cost
is divided by its mean (one if all costs are zero), a positive scaling that
preserves the exact OT optimization objective but can affect floating-point
behavior/ties. Sinkhorn's `reg` is relative to this scaled cost, **not** an
absolute latent-coordinate distance. Sinkhorn adds entropy and is a different
method variant; it is not guaranteed to be faster or more accurate. Small `reg`
can still cause slow convergence. The implementation uses POT's
[log-domain solver](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.sinkhorn_log).
The public `tolerance` bounds the L1 residual of each marginal, with POT's internal
L2 stopping threshold tightened to ensure consistent checks.

`max_cost_entries=25_000_000` checks the source-by-target matrix size before
allocation. At the default bound, the float64 cost and plan alone need about
381 MiB; solver workspaces require more. This is a guard, not an out-of-core OT
solver or a total memory cap. Explicitly raise the bound only when RAM permits,
or reduce reference cells upstream using a predeclared sampling protocol.
Attention similarity is computed in query batches but still spans all references.
For large reference sets, lower `batch_size` as well.

`solve_transport` returns `(plan, diagnostics)` for custom pipelines. Successful
model calls expose `last_ot_diagnostics_`. Failed diagnostics are on the exception;
the model's previous diagnostics are cleared when a new OT solve begins.

## Prediction variants and leakage prevention

```python
pred = model.predict(train, target_type, keys, ratio=0.05,
                     ot_config=exact, attention_mode="cellwise",
                     weighting="signed", mapping="argmax",
                     batch_size=1024, sample=False)
```

- `ratio` is in `(0, 1]`; top-k is `ceil(number_of_source_controls * ratio)`.
- `attention_mode="cellwise"` uses each target cell's own neighbors and weights.
  `"legacy_first"` broadcasts the first target's delta and emits a warning.
- `weighting="signed"` retains top-k cosine/sum weights. A zero or nearly
  cancelling denominator uses uniform selected-neighbor weights with a warning.
  Negative similarities can still produce negative weights. `"softmax"` is an
  explicitly different weighting scheme with positive `temperature` (default 1).
- `mapping="argmax"` retains historical maximal-plan matching.
  `"barycentric"` transfers transport-weighted source perturbed latents.
- `sample=False` uses posterior means; `True` samples the posterior.
- `layer` selects an expression layer. If omitted at inference, the layer used
  at training is reused. A provided override must have the same preprocessing
  scale; it is the caller's responsibility to ensure this.

Prediction requires target controls, source controls, and source perturbed cells.
It rejects target perturbed cells in its inputs. This check cannot detect leakage
from previous training/preprocessing: remove held-out targets **before** those
steps. The method predicts a known perturbation in a held-out cell type; it does
not infer an unseen perturbation without any source response information. Do
not rank it jointly with methods allowed target perturbation inputs without
separating the information-access settings.

`cross_cell_predict` uses paired source type means without OT. `n_top=None`
retains L1-normalized all-type attention; integer `n_top` uses top-k. Unpaired
source types are skipped with a warning. Historical random type balancing has
been removed, so `legacy_first` alone does not reproduce that entire old path.
`ot_predict` retains direct barycentric transport over all control cells to
source perturbed cells, without the attention stage.

Metadata and observation/feature names are preserved. Prediction metadata in
`pred.uns["scpram"]` is h5ad-serializable. Inputs are not modified. Once trained
or loaded, the model checks exact gene names and order rather than only length.

## Historical reproduction and checkpoints

To compare with first-target attention, use `attention_mode="legacy_first"`,
`sample=True`, `weighting="signed"`, `mapping="argmax"`, and checked EMD. This
retains the historical formula but is **not** a promise of bitwise reproduction:
batching, float64 costs, solver tolerances, random-number consumption and tie
ordering may differ. Use the original source commit `a462429` and its environment
for strict historical replication.

```python
model.save("scpram_model.pt")
restored = SCPRAM.load("scpram_model.pt", device="cpu")
```

Checkpoints include architecture, weights, feature order, layer, history and
stopping metadata. They use PyTorch's restricted weights loader. They do not
include preprocessing transforms, optimizer state, raw data, or OT plans. Old
plain state dictionaries remain loadable with `model.load_state_dict(...)` if
the architecture matches, but lack gene-order metadata until fitted again.

## Evidence and next work

Automated tests exercise mathematical invariants and small synthetic pipelines.
They do not prove OT convergence for every dataset, a runtime improvement on
large folds, or better biological scores. The next scientific comparison should
freeze folds/seeds/information access, retain both positive and negative results,
and report legacy attention, corrected attention, and entropic variants
separately. No full benchmark is part of this release validation.
