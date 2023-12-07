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

## 
