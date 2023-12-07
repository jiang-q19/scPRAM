# scPRAM accurately predicts single-cell gene expression perturbation response based on attention mechanism

![overview_scpram](https://github.com/jiang-q19/scPRAM/blob/main/overview_scpram.png)

## Installation

It's prefered to create a new environment for scPRAM

```
conda create -n scPRAM python==3.8
conda activate scPRAM
```

scPRAM is available on PyPI, and could be installed using

```
# CUDA 11.6
pip install scpram --extra-index-url https://download.pytorch.org/whl/cu116

# CUDA 11.3
pip install scpram --extra-index-url https://download.pytorch.org/whl/cu113

# CUDA 10.2
pip install scpram --extra-index-url https://download.pytorch.org/whl/cu102

# CPU only
pip install scpram --extra-index-url https://download.pytorch.org/whl/cpu
```

Installation via Github is also provided

```
git clone https://github.com/jiang-q19/scPRAM
cd scPRAM

# CUDA 11.6
pip install scpram.whl --extra-index-url https://download.pytorch.org/whl/cu116

# CUDA 11.3
pip install scpram.whl --extra-index-url https://download.pytorch.org/whl/cu113

# CUDA 10.2
pip install scpram.whl --extra-index-url https://download.pytorch.org/whl/cu102

# CPU only
pip install scpram.whl --extra-index-url https://download.pytorch.org/whl/cpu
```

This process will take approximately 5 to 10 minutes, depending on the user's computer device and internet connectivition.

## Quick Start

Out-of-sample prediction across cell types is demonstrated here. scPRAM can be easily used through three steps: data preprocessing, model training, prediction and evaluation. See the [tutorial]([scPRAM/Tutorial/PBMC_cross_celltype_predict.ipynb at main · jiang-q19/scPRAM (github.com)](https://github.com/jiang-q19/scPRAM/blob/main/Tutorial/PBMC_cross_celltype_predict.ipynb)) for a pipeline demonstration using the PBMC data set as an example.

#### 1. Data preprocessing

If your perturbation dataset has already undergone quality control and preprocessing with Scanpy, please disregard this step. Otherwise, you can perform preprocessing using the following code. The specific parameters can be adjusted according to the actual situation of your dataset.

```python
from scpram.data_process import adata_process
adata = adata_process(adata, min_genes=200, min_cells=10, n_top_genes=6000)
```

#### 2. Model training

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

#### 3. Predicting and evaluating

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


