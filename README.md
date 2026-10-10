# scPRAM accurately predicts single-cell gene expression perturbation response based on attention mechanism

![overview_scpram](https://github.com/jiang-q19/scPRAM/blob/main/overview_scpram.png)

## Version 0.1.0

This release adds checked OT solvers, validation early stopping, batched sparse
input handling, per-cell attention, and model checkpoints. It retains the VAE
architecture and training loss. Prediction defaults have changed: posterior means
and per-cell attention replace sampled latents and first-target-cell attention.
These engineering checks do **not** establish improved biological accuracy.
See the [migration and API guide](docs/scpram_0.1.0_migration_and_api.md) and
[Chinese change and validation notes](docs/scpram_0.1.0_优化与验证说明_20261010.md).

## Installation

Use Python 3.10 or newer in a separate environment. Install an appropriate
PyTorch build for your CPU/CUDA environment, then install the source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[plot]'
```

For a fresh checkout:

```bash
git clone https://github.com/jiang-q19/scPRAM.git
cd scPRAM
git switch release/0.1.0
python -m pip install -e '.[plot]'
```

The `plot` extra supports the evaluation plots; `notebook` installs Jupyter and
`dev` installs testing/build tools. Do not install the historical `scpram.whl`
in the repository to obtain this version. A GitHub update does not publish a new
PyPI package. Original notebooks describe the historical API and settings.

## Quick Start

Out-of-sample prediction across cell types is demonstrated here. scPRAM can be easily used through three steps: data preprocessing, model training, prediction and evaluation. See the [tutorial](https://github.com/jiang-q19/scPRAM/blob/main/Tutorial/PBMC_cross_celltype_predict.ipynb) for a pipeline demonstration using the PBMC data set as an example.

### 1. Data preprocessing

If your perturbation dataset has already undergone quality control and preprocessing with Scanpy, please disregard this step. Otherwise, you can perform preprocessing using the following code. The specific parameters can be adjusted according to the actual situation of your dataset.

```python
from scpram.data_process import adata_process
adata = adata_process(adata, min_genes=200, min_cells=10, n_top_genes=6000)
```

### 2. Model training

Before starting the training, you need to determine the specific values in the `key_dic` based on your dataset. After holding out the perturbed data for the target type, the remaining data will be used as the training set.

```python
from scpram import models
model = models.SCPRAM(input_dim=adata.n_vars, device='cuda:0')
model = model.to(model.device)
# key_dic varies with the adata
key_dic = {'condition_key': 'condition',
           'cell_type_key': 'cell_type',
           'ctrl_key': 'control',
           'stim_key': 'stimulated',
           'pred_key': 'predict',
           }
cell_to_pred = 'CD4T'
# The training set does not contain the type of data to be predicted after the perturbation
train = adata[~((adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
               (adata.obs[key_dic['condition_key']] == key_dic['stim_key']))]
model.train_SCPRAM(train, epochs=100)
```

### 3. Predicting and evaluating

After completing the training, out-of-sample predictions can be made, and the output results represent the predicted perturbation response. We combine the predicted response, actual response, and data before perturbation for performance evaluation.

```python
from scpram import evaluate
pred = model.predict(train_adata=train,
                     cell_to_pred=cell_to_pred,
                     key_dic=key_dic,
                     ratio=0.005)  # The ratio need to vary with the size of dataset

ground_truth = adata[(adata.obs[key_dic['cell_type_key']] == cell_to_pred)]
eval_adata = ground_truth.concatenate(pred)
evaluate.evaluate_adata(eval_adata=eval_adata, 
                        cell_type=cell_to_pred, 
                        key_dic=key_dic
                        )
```



## Validation early stopping and explicit OT settings

Split only the permitted training data. Held-out target perturbations must stay
outside training, validation, preprocessing fit, and prediction inputs.

```python
from scpram import SCPRAM, OTConfig

model = SCPRAM(input_dim=train.n_vars, seed=42, device="cpu")
history = model.train_SCPRAM(
    train, epochs=300, batch_size=128,
    validation_fraction=0.1, patience=20, min_delta=1e-4, seed=42,
)
pred = model.predict(
    train, cell_to_pred, key_dic,
    ot_config=OTConfig(method="emd", max_iter=1_000_000, retries=1),
    attention_mode="cellwise", batch_size=1024,
)
model.save("scpram_model.pt")
restored = SCPRAM.load("scpram_model.pt", device="cpu")
```

Early stopping monitors deterministic VAE validation loss, not perturbation
prediction accuracy. Without validation or `patience`, old fixed-epoch calls
remain valid. Full OT diagnostics are in `model.last_ot_diagnostics_`; essential
prediction settings are stored in `pred.uns["scpram"]`.

## Development

```bash
python -m pip install -e '.[dev,plot]'
python -m pytest
ruff check scpram/models.py scpram/transport.py scpram/attention.py scpram/__init__.py tests setup.py
python -m build
```
