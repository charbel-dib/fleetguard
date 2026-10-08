"""Small tabular MLP; group-aware epoch selection stays strictly inside fit data."""

import math

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch import nn

from fleetguard.comparison_splits import grouped_folds


def _preprocessor():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Select epochs on a fit-only holdout, then refit all supplied fit rows.

    Persist NumPy weights rather than live CPU/CUDA modules. Inference always uses
    CPU, including when the training device was CUDA. Missing values are imputed
    and scaled using only the rows allowed at each training stage.
    """

    def __init__(
        self,
        *,
        width=64,
        layers=2,
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_size=1024,
        max_epochs=24,
        patience=5,
        inner_folds=5,
        random_state=42,
        n_jobs=2,
        device="cpu",
    ):
        self.width = width
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.inner_folds = inner_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device

    def _network(self, dimensions):
        modules, previous = [], dimensions
        for _ in range(self.layers):
            modules.extend([nn.Linear(previous, self.width), nn.ReLU(), nn.Dropout(self.dropout)])
            previous = self.width
        modules.append(nn.Linear(previous, 1))
        return nn.Sequential(*modules)

    def _train(self, x, y, *, epochs, x_stop=None, y_stop=None):
        network = self._network(x.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        pos_weight = float((y == 0).sum() / (y == 1).sum())
        loss_function = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device, dtype=torch.float32)
        )
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        best, best_epoch, stale, history = math.inf, 0, 0, []
        for epoch in range(1, epochs + 1):
            network.train()
            order = torch.randperm(len(yt), device=self.device)
            total_loss = 0.0
            for indices in order.split(self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(network(xt[indices]).flatten(), yt[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError("MLP loss is nonfinite; no artifact will be published.")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(indices)
            record = {"epoch": epoch, "fit_loss": total_loss / len(yt)}
            if x_stop is not None:
                network.eval()
                with torch.no_grad():
                    logits = network(
                        torch.as_tensor(x_stop, dtype=torch.float32, device=self.device)
                    )
                    heldout_loss = loss_function(
                        logits.flatten(),
                        torch.as_tensor(y_stop, dtype=torch.float32, device=self.device),
                    )
                value = float(heldout_loss.cpu())
                if not math.isfinite(value):
                    raise RuntimeError("MLP early-stopping loss is nonfinite.")
                record["early_stopping_loss"] = value
                if value < best - 1e-6:
                    best, best_epoch, stale = value, epoch, 0
                else:
                    stale += 1
            history.append(record)
            if x_stop is not None and stale >= self.patience:
                break
        return network, best_epoch if x_stop is not None else epochs, history

    def fit(self, X, y):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("MLP device must be cpu or cuda.")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable. Use device='cpu'.")
        if (
            any(
                type(x) is not int or x < 1
                for x in (
                    self.width,
                    self.layers,
                    self.batch_size,
                    self.max_epochs,
                    self.patience,
                    self.n_jobs,
                )
            )
            or self.inner_folds < 2
        ):
            raise ValueError("MLP dimensions and budgets must be positive; inner_folds >= 2.")
        if not 0 <= self.dropout < 1 or not math.isfinite(self.dropout):
            raise ValueError("MLP dropout must lie in [0, 1).")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("MLP learning rate must be positive and finite.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("MLP weight decay must be finite and nonnegative.")
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        target = pd.Series(np.asarray(y), index=frame.index)
        subfit, stop = grouped_folds(
            frame, target, n_splits=self.inner_folds, seed=self.random_state
        )[0]
        self.epoch_roles_ = pd.DataFrame(
            {
                "local_row": np.arange(len(frame)),
                "feature_group": pd.util.hash_pandas_object(frame, index=False).to_numpy(),
                "role": np.where(np.isin(np.arange(len(frame)), stop), "early_stopping", "subfit"),
            }
        )
        preprocessor = _preprocessor()
        x_subfit = preprocessor.fit_transform(frame.iloc[subfit]).astype(np.float32)
        x_stop = preprocessor.transform(frame.iloc[stop]).astype(np.float32)
        previous_threads = torch.get_num_threads()
        devices = [torch.cuda.current_device()] if self.device == "cuda" else []
        try:
            torch.set_num_threads(self.n_jobs)
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.random_state)
                _, self.selected_epochs_, self.history_ = self._train(
                    x_subfit,
                    target.iloc[subfit].to_numpy(),
                    epochs=self.max_epochs,
                    x_stop=x_stop,
                    y_stop=target.iloc[stop].to_numpy(),
                )
                self.preprocessor_ = _preprocessor()
                x_full = self.preprocessor_.fit_transform(frame).astype(np.float32)
                torch.manual_seed(self.random_state)
                network, _, self.refit_history_ = self._train(
                    x_full, target.to_numpy(), epochs=self.selected_epochs_
                )
                self.state_dict_ = {
                    name: tensor.detach().cpu().numpy().copy()
                    for name, tensor in network.state_dict().items()
                }
        finally:
            torch.set_num_threads(previous_threads)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns)
        self.network_features_ = x_full.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "state_dict_")
        x = self.preprocessor_.transform(X).astype(np.float32)
        # Network construction consumes RNG; restore it so inference does not change training RNG.
        with torch.random.fork_rng(devices=[]):
            network = self._network(self.network_features_)
        network.load_state_dict(
            {name: torch.from_numpy(value) for name, value in self.state_dict_.items()}
        )
        network.eval()
        values = []
        with torch.no_grad():
            for batch in torch.from_numpy(x).split(self.batch_size):
                values.append(torch.sigmoid(network(batch).flatten()).numpy())
        positive = np.concatenate(values)
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
