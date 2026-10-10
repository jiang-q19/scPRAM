from copy import deepcopy
from unittest.mock import patch

import anndata as ad
import numpy as np
import pytest
from scipy import sparse
import torch

from scpram import OTConfig, SCPRAM, models
from scpram.dataset import AnnDataSet


def test_legacy_import_and_positional_training(model, data):
    assert models.SCPRAM is SCPRAM
    history = model.train_SCPRAM(data, 2, 7, 5e-4, verbose=False)
    assert len(history) == 2
    assert np.isfinite(history[-1]["train_loss"])
    assert not model.training


def test_early_stop_restores_actual_best_weights(model, data):
    snapshots = []
    losses = iter([3.0, 2.0, 2.5, 2.6, 1.0])

    def validation(_):
        snapshots.append(deepcopy(model.state_dict()))
        return next(losses)

    with patch.object(model, "_validation_loss", side_effect=validation):
        history = model.train_SCPRAM(
            data,
            epochs=10,
            batch_size=8,
            validation_adata=data[:5],
            patience=2,
            verbose=False,
            seed=3,
        )
    assert len(history) == 4 and model.best_epoch_ == 2 and model.stopped_early_
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, snapshots[1][key], rtol=0, atol=0)


def test_validation_deterministic_and_sample_weighted(model, data):
    a = model._validation_loss(model._loader(data, 7))
    b = model._validation_loss(model._loader(data, 30))
    assert a == pytest.approx(b, rel=1e-6)
    assert a == model._validation_loss(model._loader(data, 7))


def test_sparse_float64_training_and_inference(model, data):
    original = data.X.copy()
    data.X = sparse.csr_matrix(data.X.astype(np.float64))
    dataset = AnnDataSet(data)
    assert sparse.issparse(dataset.data)
    assert dataset[0].dtype == np.float32
    # Entire matrices must never be materialized via AnnData.to_df.
    with patch.object(ad.AnnData, "to_df", side_effect=AssertionError("dense conversion")):
        model.fit(data, epochs=2, validation_fraction=0.2, patience=2, seed=5, verbose=False)
        latent = model.get_latent_adata(data, batch_size=7)
    np.testing.assert_array_equal(data.X.toarray(), original)
    np.testing.assert_allclose(latent.X, model.get_latent_adata(data, batch_size=30).X, atol=1e-6)


def test_seed_reproduces_training_and_initialization(data):
    a = SCPRAM(5, 3, 12, device="cpu", seed=1)
    b = SCPRAM(5, 3, 12, device="cpu", seed=1)
    options = dict(epochs=3, seed=2, validation_fraction=0.2, patience=2, verbose=False)
    a.fit(data, **options)
    b.fit(data, **options)
    assert a.history_ == b.history_
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("method", ["predict", "ot_predict", "cross_cell_predict"])
def test_prediction_interfaces_preserve_metadata_and_inputs(model, data, keys, method, tmp_path):
    obs = data.obs.copy(deep=True)
    model.train()
    output = getattr(model, method)(data, "target", keys, batch_size=3)
    assert output.shape == (6, 5)
    assert list(output.obs_names) == list(data.obs_names[-6:])
    assert list(output.var_names) == list(data.var_names)
    assert (output.obs.condition == "pred").all()
    assert np.isfinite(output.X).all()
    assert model.training
    assert data.obs.equals(obs)
    output.write_h5ad(tmp_path / "prediction.h5ad")
    restored = ad.read_h5ad(tmp_path / "prediction.h5ad")
    assert restored.uns["scpram"]["version"] == "0.1.0"


def test_prediction_is_deterministic_and_target_order_independent(model, data, keys):
    a = model.predict(data, "target", keys)
    b = model.predict(data, "target", keys)
    permuted = ad.concat([data[:24], data[24:][::-1]])
    c = model.predict(permuted, "target", keys)
    np.testing.assert_allclose(a.X, b.X, rtol=0, atol=0)
    np.testing.assert_allclose(a.X, c.X[::-1], atol=1e-6)


def test_explicit_sinkhorn_barycentric_variant(model, data, keys):
    result = model.predict(
        data,
        "target",
        keys,
        ot_config=OTConfig(method="sinkhorn"),
        mapping="barycentric",
        weighting="softmax",
    )
    assert result.uns["scpram"]["ot_method"] == "sinkhorn"
    assert model.last_ot_diagnostics_["converged"]


def test_checkpoint_gene_order_and_layer(model, data, keys, tmp_path):
    data.layers["expression"] = data.X.copy()
    model.fit(data, epochs=2, layer="expression", verbose=False)
    before = model.predict(data, "target", keys)
    model.save(tmp_path / "model.pt")
    loaded = SCPRAM.load(tmp_path / "model.pt")
    after = loaded.predict(data, "target", keys)
    np.testing.assert_array_equal(before.X, after.X)
    assert loaded.history_ == model.history_ and loaded.layer_ == "expression"
    with pytest.raises(ValueError, match="order"):
        loaded.get_latent_adata(data[:, ::-1])


@pytest.mark.parametrize("method", ["predict", "ot_predict", "cross_cell_predict"])
def test_target_perturbation_leakage_is_rejected(model, data, keys, method):
    data.obs.loc[data.obs_names[-1], "condition"] = "stim"
    with pytest.raises(ValueError, match="held-out"):
        getattr(model, method)(data, "target", keys)


def test_empty_groups_and_invalid_parameters(model, data, keys):
    with pytest.raises(ValueError, match="source"):
        model.predict(data[24:], "target", keys)
    with pytest.raises(ValueError, match="ratio"):
        model.predict(data, "target", keys, ratio=0)
    with pytest.raises(ValueError, match="validation"):
        model.fit(data, patience=2, verbose=False)
    data.X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.get_latent_adata(data)


def test_device_property_tracks_to(model):
    model.to("cpu")
    assert model.device == next(model.parameters()).device


def test_cross_cell_unpaired_types_warn_and_no_pairs_fail(model, data, keys):
    # Remove source_b stimulation; source_a remains a valid reference pair.
    subset = data[~((data.obs.cell_type == "source_b") & (data.obs.condition == "stim"))]
    with pytest.warns(UserWarning, match="unpaired"):
        result = model.cross_cell_predict(subset, "target", keys, n_top=1)
    assert result.shape == (6, 5)
    subset = data[
        ~(
            ((data.obs.cell_type == "source_b") & (data.obs.condition == "stim"))
            | ((data.obs.cell_type == "source_a") & (data.obs.condition == "ctrl"))
        )
    ]
    with pytest.raises(ValueError, match="both control"):
        model.cross_cell_predict(subset, "target", keys)


def test_nonfinite_loss_does_not_update_weights(model, data):
    before = deepcopy(model.state_dict())
    with patch.object(
        model, "get_loss", return_value=(torch.tensor([float("nan")]), torch.zeros(1))
    ):
        with pytest.raises(FloatingPointError, match="training loss"):
            model.fit(data, epochs=1, verbose=False)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_training_epoch_loss_weights_short_final_batch(model, data):
    # Keep weights fixed, remove noise, and use deterministic evaluation forwards.
    model.noise_rate = 0.0
    with patch.object(
        model,
        "encode",
        side_effect=lambda x, sample=True: (
            model.encoder(x).chunk(2, dim=1)[0],
            *model.encoder(x).chunk(2, dim=1),
        ),
    ):
        expected = model._validation_loss(model._loader(data, 30))
        with patch("torch.optim.Adam.step"):
            history = model.train_SCPRAM(data, epochs=1, batch_size=7, verbose=False)
    assert history[0]["train_loss"] == pytest.approx(expected, rel=1e-6)
