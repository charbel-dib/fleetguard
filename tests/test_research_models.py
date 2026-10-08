from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from fleetguard.artifacts import load_model, positive_scores, save_model
from fleetguard.io import read_json, write_json


@pytest.fixture
def research():
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")


def fixture_data():
    x, y = make_classification(
        n_samples=400, n_features=8, n_informative=5, weights=[0.9, 0.1], random_state=42
    )
    features = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(8)])
    features.iloc[::7, 0] = np.nan
    # Duplicate observations must remain together during epoch selection.
    features = pd.concat([features, features.iloc[:12]], ignore_index=True)
    return features, pd.Series(np.r_[y, y[:12]])


def test_mlp_reproducible_portable_and_preserves_rng(tmp_path, research):
    import torch

    from fleetguard.torch_model import TorchMLPClassifier

    x, y = fixture_data()
    torch.manual_seed(123)
    rng, threads = torch.random.get_rng_state().clone(), torch.get_num_threads()
    kwargs = dict(width=8, max_epochs=3, patience=1, inner_folds=2, n_jobs=1, random_state=42)
    model = TorchMLPClassifier(**kwargs).fit(x, y)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert threads == torch.get_num_threads()
    assert model.epoch_roles_.groupby("feature_group")["role"].nunique().max() == 1
    assert 1 <= model.selected_epochs_ <= 3
    expected = positive_scores(model, x.iloc[:10])
    assert torch.equal(rng, torch.random.get_rng_state())
    repeated = TorchMLPClassifier(**kwargs).fit(x, y)
    np.testing.assert_allclose(expected, positive_scores(repeated, x.iloc[:10]), rtol=0, atol=0)
    save_model(tmp_path / "model", model, metadata={"torch_version": torch.__version__})
    loaded, _ = load_model(tmp_path / "model")
    np.testing.assert_allclose(expected, positive_scores(loaded, x.iloc[:10]), rtol=0, atol=0)
    metadata = read_json(tmp_path / "model/metadata.json")
    metadata["torch_version"] = "0.0.invalid"
    write_json(tmp_path / "model/metadata.json", metadata)
    with pytest.raises(ValueError, match="PyTorch versions differ"):
        load_model(tmp_path / "model")


def test_threshold_labels_cannot_change_mlp_or_calibrator(tmp_path, research):
    from fleetguard.optimization import _roles, fit_candidate
    from fleetguard.optimization_smoke import research_smoke_config

    x, y = fixture_data()
    config = research_smoke_config(tmp_path)
    roles, _ = _roles(x, y, config, 43)
    params = dict(
        width=8, layers=1, dropout=0.1, learning_rate=0.001, weight_decay=0.0001, batch_size=64
    )
    original = fit_candidate(x, y, roles, "mlp", params, config)
    changed = y.copy()
    changed.iloc[roles["threshold"]] = 1 - changed.iloc[roles["threshold"]]
    altered = fit_candidate(x, changed, roles, "mlp", params, config)
    for name in original["raw_model"].state_dict_:
        np.testing.assert_array_equal(
            original["raw_model"].state_dict_[name], altered["raw_model"].state_dict_[name]
        )
    np.testing.assert_allclose(
        positive_scores(original["model"], x), positive_scores(altered["model"], x), rtol=0, atol=0
    )
    fit_rows = set(roles["fit"])
    epoch = original["raw_model"].epoch_roles_["local_row"].to_numpy()
    assert set(original["fit_rows"][epoch]) <= fit_rows


def test_oversampling_and_preprocessing_use_fit_rows_only(tmp_path, research):
    from fleetguard.optimization import _roles, fit_candidate
    from fleetguard.optimization_smoke import research_smoke_config

    x, y = fixture_data()
    config = research_smoke_config(tmp_path)
    roles, _ = _roles(x, y, config, 43)
    bundle = fit_candidate(x, y, roles, "logistic_oversampled", {}, config)
    pipeline = bundle["model"]
    sampler = pipeline.named_steps["oversampler"]
    assert sampler.sample_indices_.max() < len(roles["fit"])
    np.testing.assert_allclose(
        pipeline.named_steps["imputer"].statistics_,
        np.nanmedian(x.iloc[roles["fit"]].to_numpy(), axis=0),
    )
    assert len(positive_scores(pipeline, x.iloc[roles["threshold"]])) == len(roles["threshold"])


def test_learning_subsets_are_nested_whole_groups(research):
    from fleetguard.optimization import subset_fit

    x, y = fixture_data()
    small = subset_fit(x, y, 0.25, seed=42)
    large = subset_fit(x, y, 0.5, seed=42)
    assert set(small) < set(large)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    assert set(groups[small]).isdisjoint(groups[np.setdiff1d(np.arange(len(x)), small)])
    assert set(y.iloc[small]) == {0, 1}


def test_cuda_xgboost_fits_on_gpu_then_predicts_and_reloads_on_cpu(tmp_path, monkeypatch, research):
    import json
    import warnings

    import torch
    from xgboost import XGBClassifier

    from fleetguard.comparison import fit_and_tune
    from fleetguard.optimization import _roles, fit_candidate
    from fleetguard.optimization_models import baseline_xgb
    from fleetguard.optimization_smoke import research_smoke_config

    if not torch.cuda.is_available():
        pytest.skip("CUDA regression requires an NVIDIA GPU")
    config = replace(research_smoke_config(tmp_path), device="cuda")
    x, y = fixture_data()
    roles, _ = _roles(x, y, config, 43)
    original_fit = XGBClassifier.fit
    gpu_fits, tree_dumps = [], []

    def record_fit(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        gpu_fits.append(
            json.loads(self.get_booster().save_config())["learner"]["generic_param"]["device"]
        )
        tree_dumps.append(self.get_booster().get_dump())
        return result

    monkeypatch.setattr(XGBClassifier, "fit", record_fit)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        optimized = fit_candidate(x, y, roles, "xgboost", baseline_xgb(config), config)
        compared = fit_and_tune(
            x, y, replace(config.comparison, models=("xgboost",), device="cuda"), roles
        )["xgboost"]
        for index, bundle in enumerate((optimized, compared)):
            model = bundle["model"]
            classifier = model.named_steps["classifier"]
            assert classifier.device == "cpu"
            assert classifier.get_booster().get_dump() == tree_dumps[index]
            expected = positive_scores(model, x)
            destination = tmp_path / str(index)
            save_model(destination, model, metadata={})
            loaded, _ = load_model(destination)
            assert loaded.named_steps["classifier"].device == "cpu"
            np.testing.assert_array_equal(expected, positive_scores(loaded, x))
    assert gpu_fits == ["cuda:0", "cuda:0"]
    assert not any("mismatched devices" in str(warning.message) for warning in observed)


def test_optimization_config_rejects_unbounded_or_ambiguous_settings(tmp_path, research):
    from fleetguard.optimization_smoke import research_smoke_config

    config = research_smoke_config(tmp_path)
    for change in (
        {"xgb_trials": 0},
        {"mlp_max_epochs": 0},
        {"device": "auto"},
        {"learning_fractions": (0.5,)},
        {"learning_fractions": (1.0, 0.5)},
        {"stability_seeds": (43, 43)},
    ):
        with pytest.raises(ValueError):
            replace(config, **change)
