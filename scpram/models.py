"""Variational autoencoder and perturbation transfer interfaces."""

from copy import deepcopy
from pathlib import Path
import warnings

import anndata as ad
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .attention import attention_delta
from .dataset import AnnDataSet
from .transport import solve_transport
from .version import __version__


class SCPRAM(nn.Module):
    """scPRAM VAE. Existing constructor and training calls remain supported.

    Seed before construction (or pass ``seed``) to reproduce initialization.
    Prediction uses posterior means by default; ``sample=True`` restores
    stochastic latent draws. The decoder and training objective are unchanged.
    """

    def __init__(
        self,
        input_dim=6998,
        latent_dim=100,
        hidden_dim=1000,
        noise_rate=0.1,
        kl_weight=5e-4,
        device=None,
        *,
        seed=None,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("latent_dim", latent_dim),
            ("hidden_dim", hidden_dim),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(noise_rate) or noise_rate < 0:
            raise ValueError("noise_rate must be finite and nonnegative.")
        if not np.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("kl_weight must be finite and nonnegative.")
        if seed is not None:
            torch.manual_seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.noise_rate = noise_rate
        self.kl_weight = kl_weight
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
        )
        self.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.history_ = []
        self.feature_names_ = None
        self.layer_ = None
        self.last_ot_diagnostics_ = None

    @property
    def device(self):
        """Actual parameter device, including after model.to(...)."""
        return next(self.parameters()).device

    def encode(self, x, sample=True):
        mu, logvar = torch.chunk(self.encoder(x), 2, dim=1)
        z = self.reparameterize(mu, logvar) if sample else mu
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        noisy = x + torch.randn_like(x) * self.noise_rate if self.training else x
        z, mu, logvar = self.encode(noisy, sample=self.training)
        x_hat = self.decode(z)
        loss_kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=1)
        loss_rec = (x - x_hat).square().sum(dim=1)
        return x_hat, loss_rec, loss_kl

    def get_loss(self, x):
        _, loss_rec, loss_kl = self.forward(x)
        return loss_rec, loss_kl

    def _validate_adata(self, adata):
        if adata.n_obs == 0 or adata.n_vars != self.input_dim:
            raise ValueError(f"Expected nonempty AnnData with {self.input_dim} genes.")
        if not adata.var_names.is_unique:
            raise ValueError("Gene names must be unique.")
        if self.feature_names_ is not None and list(adata.var_names) != self.feature_names_:
            raise ValueError("Gene names/order differ from training; align var_names first.")

    def _loader(self, adata, batch_size, layer=None, shuffle=False, generator=None):
        self._validate_adata(adata)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        layer = self.layer_ if layer is None else layer
        return DataLoader(
            AnnDataSet(adata, layer=layer),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            drop_last=False,
        )

    def _input(self, x):
        x = x.to(device=self.device, dtype=next(self.parameters()).dtype)
        if not torch.isfinite(x).all():
            raise ValueError("Expression input contains NaN or infinity.")
        return x

    @torch.no_grad()
    def get_latent_adata(self, adata, *, batch_size=1024, sample=False, layer=None):
        loader = self._loader(adata, batch_size, layer)
        was_training = self.training
        self.eval()
        try:
            latent = [self.encode(self._input(x), sample=sample)[0].cpu().numpy() for x in loader]
        finally:
            self.train(was_training)
        result = np.concatenate(latent)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite latent coordinates.")
        return ad.AnnData(X=result, obs=adata.obs.copy())

    def train_SCPRAM(
        self,
        train_adata,
        epochs=100,
        batch_size=128,
        lr=5e-4,
        *,
        validation_adata=None,
        validation_fraction=0.0,
        patience=None,
        min_delta=0.0,
        restore_best=True,
        seed=None,
        layer=None,
        weight_decay=1e-5,
        grad_clip=10.0,
        verbose=True,
    ):
        """Train with sample-weighted epoch metrics and optional validation early stopping.

        Validation uses clean input and posterior means, so it is deterministic.
        It measures VAE reconstruction/KL, not perturbation prediction quality.
        Split only permitted training data; never pass held-out target perturbations.
        ``patience=None`` retains fixed-epoch training. Returns epoch history.
        """
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if patience is not None and (not isinstance(patience, int) or patience < 1):
            raise ValueError("patience must be a positive integer or None.")
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative.")
        for name, value in (("lr", lr), ("grad_clip", grad_clip)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1).")
        if validation_adata is not None and validation_fraction:
            raise ValueError("Choose validation_adata or validation_fraction, not both.")
        self._validate_adata(train_adata)
        if validation_fraction:
            n_val = max(1, int(train_adata.n_obs * validation_fraction))
            if n_val >= train_adata.n_obs:
                raise ValueError("The validation split leaves no training cells.")
            indices = np.random.default_rng(seed).permutation(train_adata.n_obs)
            validation_adata = train_adata[indices[:n_val]]
            train_adata = train_adata[indices[n_val:]]
        if patience is not None and validation_adata is None:
            raise ValueError("Early stopping requires a validation set or validation_fraction.")
        self.feature_names_ = list(train_adata.var_names)
        self.layer_ = layer
        generator = None
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
        loader = self._loader(train_adata, batch_size, layer, True, generator)
        val_loader = (
            self._loader(validation_adata, batch_size, layer)
            if validation_adata is not None
            else None
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.history_ = []
        self.best_epoch_ = None
        self.stopped_early_ = False
        best_loss, stopping_best, stale, best_state = float("inf"), float("inf"), 0, None
        pbar = tqdm(range(1, epochs + 1), disable=not verbose)
        for epoch in pbar:
            self.train()
            total = np.zeros(3)
            for x in loader:
                x = self._input(x)
                rec, kl = self.get_loss(x)
                loss = (0.5 * rec + 0.5 * self.kl_weight * kl).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
                total += np.array([loss.item() * len(x), rec.sum().item(), kl.sum().item()])
            total /= len(loader.dataset)
            record = dict(
                epoch=epoch,
                train_loss=float(total[0]),
                reconstruction_loss=float(total[1]),
                kl_loss=float(total[2]),
            )
            if val_loader is not None:
                val_loss = self._validation_loss(val_loader)
                record["val_loss"] = val_loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.best_epoch_ = epoch
                    if restore_best:
                        best_state = {
                            k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                        }
                if val_loss < stopping_best - min_delta:
                    stopping_best, stale = val_loss, 0
                else:
                    stale += 1
            self.history_.append(record)
            pbar.set_postfix({k: v for k, v in record.items() if k in {"train_loss", "val_loss"}})
            if patience is not None and stale >= patience:
                self.stopped_early_ = True
                break
        pbar.close()
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        return deepcopy(self.history_)

    def fit(self, train_adata, **kwargs):
        """Chainable alias for train_SCPRAM; metrics are in history_."""
        self.train_SCPRAM(train_adata, **kwargs)
        return self

    @torch.no_grad()
    def _validation_loss(self, loader):
        self.eval()
        total = 0.0
        for x in loader:
            rec, kl = self.get_loss(self._input(x))
            total += (0.5 * rec + 0.5 * self.kl_weight * kl).sum().item()
        value = total / len(loader.dataset)
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite validation loss.")
        return value

    def _prediction_groups(self, adata, cell_to_pred, keys):
        self._validate_adata(adata)
        required = {"condition_key", "cell_type_key", "ctrl_key", "stim_key", "pred_key"}
        if not required.issubset(keys):
            raise ValueError(f"key_dic is missing {sorted(required - keys.keys())}.")
        for name in ("condition_key", "cell_type_key"):
            if keys[name] not in adata.obs:
                raise ValueError(f"Missing obs column: {keys[name]}.")
            if adata.obs[keys[name]].isna().any():
                raise ValueError(f"Missing labels in obs column: {keys[name]}.")
        if keys["ctrl_key"] == keys["stim_key"]:
            raise ValueError("Control and stimulated labels must differ.")
        condition = adata.obs[keys["condition_key"]]
        target = adata.obs[keys["cell_type_key"]] == cell_to_pred
        if (target & (condition == keys["stim_key"])).any():
            raise ValueError("Remove held-out target perturbed cells from prediction inputs.")
        groups = (
            adata[target & (condition == keys["ctrl_key"])],
            adata[~target & (condition == keys["ctrl_key"])],
            adata[~target & (condition == keys["stim_key"])],
        )
        if any(x.n_obs == 0 for x in groups):
            raise ValueError("Need target controls, source controls, and source perturbed cells.")
        return groups

    @torch.no_grad()
    def _prediction_output(self, latent, controls, keys, batch_size, metadata):
        was_training = self.training
        self.eval()
        try:
            output = [
                self.decode(self._input(torch.as_tensor(latent[i : i + batch_size]))).cpu().numpy()
                for i in range(0, len(latent), batch_size)
            ]
        finally:
            self.train(was_training)
        output = np.concatenate(output)
        if not np.isfinite(output).all():
            raise FloatingPointError("Nonfinite decoded prediction.")
        result = ad.AnnData(X=output, obs=controls.obs.copy(), var=controls.var.copy())
        result.obs[keys["condition_key"]] = keys["pred_key"]
        result.uns["scpram"] = {"version": __version__, **metadata}
        return result

    def predict(
        self,
        train_adata,
        cell_to_pred,
        key_dic,
        ratio=0.05,
        *,
        ot_config=None,
        attention_mode="cellwise",
        weighting="signed",
        temperature=1.0,
        mapping="argmax",
        batch_size=1024,
        sample=False,
        layer=None,
    ):
        """Transfer source perturbations to target controls using OT and attention.

        Defaults: checked exact EMD, historical argmax matching, corrected
        per-cell attention, deterministic latent means. ``legacy_first`` and
        ``sample=True`` approximate the historical prediction path. Sinkhorn,
        barycentric mapping and softmax weights are explicit method variants.
        """
        if not np.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1].")
        if mapping not in {"argmax", "barycentric"}:
            raise ValueError("mapping must be 'argmax' or 'barycentric'.")
        target, controls, stimulated = self._prediction_groups(train_adata, cell_to_pred, key_dic)
        options = dict(batch_size=batch_size, sample=sample, layer=layer)
        ctrl = self.get_latent_adata(controls, **options).X
        stim = self.get_latent_adata(stimulated, **options).X
        self.last_ot_diagnostics_ = None
        plan, diagnostics = solve_transport(stim, ctrl, ot_config)
        self.last_ot_diagnostics_ = diagnostics
        matched = (
            stim[np.argmax(plan, axis=0)]
            if mapping == "argmax"
            else (plan.T @ stim / plan.sum(axis=0)[:, None])
        )
        query = self.get_latent_adata(target, **options).X
        delta = attention_delta(
            query,
            ctrl,
            matched - ctrl,
            int(np.ceil(len(ctrl) * ratio)),
            mode=attention_mode,
            weighting=weighting,
            temperature=temperature,
            batch_size=batch_size,
        )
        # Keep the full attempt history on the model; flat output metadata can be
        # written by AnnData to h5ad without object-array serialization failures.
        metadata = dict(
            method="ot_attention",
            ot_method=diagnostics["method"],
            ot_converged=True,
            ot_cost_scale=diagnostics["cost_scale"],
            ot_attempts=len(diagnostics["attempts"]),
            ot_config=diagnostics["config"],
            ot_marginal_error=diagnostics["attempts"][-1]["marginal_error"],
            attention_mode=attention_mode,
            weighting=weighting,
            temperature=temperature,
            mapping=mapping,
            ratio=ratio,
            sample=sample,
        )
        return self._prediction_output(query + delta, target, key_dic, batch_size, metadata)

    def ot_predict(
        self,
        adata_train,
        cell_to_pred,
        key_dic,
        *,
        ot_config=None,
        batch_size=1024,
        sample=False,
        layer=None,
    ):
        """Direct barycentric OT mapping (without attention), as in the legacy API."""
        target, _, _ = self._prediction_groups(adata_train, cell_to_pred, key_dic)
        keys = adata_train.obs[key_dic["condition_key"]]
        controls = adata_train[keys == key_dic["ctrl_key"]]
        options = dict(batch_size=batch_size, sample=sample, layer=layer)
        # Encode jointly to preserve the historical sampling order when requested.
        latent = self.get_latent_adata(adata_train, **options).X
        ctrl = latent[np.asarray(keys == key_dic["ctrl_key"])]
        stim = latent[np.asarray(keys == key_dic["stim_key"])]
        self.last_ot_diagnostics_ = None
        plan, diagnostics = solve_transport(ctrl, stim, ot_config)
        self.last_ot_diagnostics_ = diagnostics
        mapped = plan @ stim / plan.sum(axis=1)[:, None]
        selected = np.asarray(controls.obs[key_dic["cell_type_key"]] == cell_to_pred)
        return self._prediction_output(
            mapped[selected],
            target,
            key_dic,
            batch_size,
            dict(
                method="direct_ot",
                ot_method=diagnostics["method"],
                ot_config=diagnostics["config"],
                sample=sample,
            ),
        )

    def cross_cell_predict(
        self,
        train_adata,
        cell_to_pred,
        key_dic,
        n_top=None,
        *,
        attention_mode="cellwise",
        weighting="signed",
        temperature=1.0,
        batch_size=1024,
        sample=False,
        layer=None,
    ):
        """Transfer per-type mean deltas without OT; uses paired source types only.

        Means are computed directly, without random oversampling of each type.
        This removes Monte Carlo variation but differs from the old balancer path.
        """
        target, controls, stimulated = self._prediction_groups(train_adata, cell_to_pred, key_dic)
        type_key = key_dic["cell_type_key"]
        types = sorted(
            (set(controls.obs[type_key]) & set(stimulated.obs[type_key])) - {"isolated"}, key=str
        )
        if not types:
            raise ValueError("No source cell type has both control and perturbed cells.")
        unmatched = (
            (set(controls.obs[type_key]) | set(stimulated.obs[type_key]))
            - set(types)
            - {"isolated"}
        )
        if unmatched:
            warnings.warn(
                f"Skipping unpaired source cell types: {sorted(unmatched, key=str)}.",
                UserWarning,
                stacklevel=2,
            )
        options = dict(batch_size=batch_size, sample=sample, layer=layer)
        ctrl = self.get_latent_adata(controls, **options).X
        stim = self.get_latent_adata(stimulated, **options).X
        ctrl_means = np.stack(
            [ctrl[np.asarray(controls.obs[type_key] == t)].mean(0) for t in types]
        )
        stim_means = np.stack(
            [stim[np.asarray(stimulated.obs[type_key] == t)].mean(0) for t in types]
        )
        query = self.get_latent_adata(target, **options).X
        delta = attention_delta(
            query,
            ctrl_means,
            stim_means - ctrl_means,
            n_top,
            mode=attention_mode,
            weighting=weighting,
            temperature=temperature,
            batch_size=batch_size,
        )
        return self._prediction_output(
            query + delta,
            target,
            key_dic,
            batch_size,
            dict(
                method="cell_type_means",
                attention_mode=attention_mode,
                weighting=weighting,
                temperature=temperature,
                n_top=-1 if n_top is None else int(n_top),
                sample=sample,
            ),
        )

    def save(self, path):
        """Save weights, architecture, gene order and metrics (not optimizer state)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "scpram_version": __version__,
                "config": {
                    k: getattr(self, k)
                    for k in ("input_dim", "latent_dim", "hidden_dim", "noise_rate", "kl_weight")
                },
                "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                "feature_names": self.feature_names_,
                "layer": self.layer_,
                "history": self.history_,
                "best_epoch": getattr(self, "best_epoch_", None),
                "stopped_early": getattr(self, "stopped_early_", False),
            },
            path,
        )

    @classmethod
    def load(cls, path, *, device="cpu"):
        """Load an scPRAM checkpoint using PyTorch's restricted weights loader."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported scPRAM checkpoint format.")
        model = cls(**checkpoint["config"], device=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.feature_names_ = checkpoint["feature_names"]
        model.layer_ = checkpoint["layer"]
        model.history_ = checkpoint["history"]
        model.best_epoch_ = checkpoint["best_epoch"]
        model.stopped_early_ = checkpoint["stopped_early"]
        model.eval()
        return model
