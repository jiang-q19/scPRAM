import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from scpram import SCPRAM


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def keys():
    return dict(
        condition_key="condition",
        cell_type_key="cell_type",
        ctrl_key="ctrl",
        stim_key="stim",
        pred_key="pred",
    )


@pytest.fixture
def data():
    rng = np.random.default_rng(12)
    obs = pd.DataFrame(
        {
            "cell_type": ["source_a"] * 12 + ["source_b"] * 12 + ["target"] * 6,
            "condition": (["ctrl"] * 6 + ["stim"] * 6) * 2 + ["ctrl"] * 6,
        },
        index=[f"cell_{i}" for i in range(30)],
    )
    obs["condition"] = pd.Categorical(obs["condition"])
    return ad.AnnData(
        rng.gamma(2, 1, (30, 5)).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )


@pytest.fixture
def model():
    return SCPRAM(input_dim=5, hidden_dim=12, latent_dim=3, device="cpu", seed=3)
