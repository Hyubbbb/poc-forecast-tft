"""mlflow_logging 모듈 unit test — mock 기반.

spec: specs/mlflow-experiment-tracking.md v0.2.0

실제 MLflow 서버 접근 X. `MlflowClient` mock 으로 행동 계약만 검증.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from scripts.mlflow_logging import (
    compute_dataset_audit,
    file_sha256,
    infer_model_architecture,
    make_run_tags,
    safe_git_sha,
    system_env_params,
)


class TestMakeRunTags:
    def test_eight_standard_keys_present_when_seed_set(self):
        tags = make_run_tags(
            experiment_intent="baseline_v1",
            data_scope="cp6677_only",
            backtest_fold="single",
            model_architecture="tft",
            random_seed=42,
            is_baseline=True,
            source="supplies_cp_e52_d26",
        )
        expected = {
            "source", "experiment_intent", "data_scope", "backtest_fold",
            "git_sha", "model_architecture", "is_baseline", "random_seed",
        }
        assert expected.issubset(tags.keys()), f"missing: {expected - tags.keys()}"
        assert tags["experiment_intent"] == "baseline_v1"
        assert tags["data_scope"] == "cp6677_only"
        assert tags["is_baseline"] == "true"
        assert tags["random_seed"] == "42"
        assert "seed_not_set" not in tags

    def test_seed_none_emits_warning_and_seed_not_set_tag(self):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            tags = make_run_tags(
                experiment_intent="test", data_scope="test", random_seed=None,
            )
        assert tags["seed_not_set"] == "true"
        assert tags["random_seed"] == "unset"
        assert any("random_seed" in str(w.message) for w in ws)

    def test_all_values_are_str(self):
        """MLflow tag 계약: 모든 value 가 str."""
        tags = make_run_tags(
            experiment_intent="x", data_scope="y", random_seed=99, is_baseline=False,
        )
        assert all(isinstance(v, str) for v in tags.values())

    def test_extra_tags_merged(self):
        tags = make_run_tags(
            experiment_intent="x", data_scope="y", random_seed=1,
            extra={"custom_tag": "abc", "another": 123},
        )
        assert tags["custom_tag"] == "abc"
        assert tags["another"] == "123"

    def test_source_default_is_unset(self):
        tags = make_run_tags(experiment_intent="x", data_scope="y", random_seed=1)
        assert tags["source"] == "unset"

    def test_is_baseline_false_stringified(self):
        tags = make_run_tags(
            experiment_intent="x", data_scope="y", random_seed=1, is_baseline=False,
        )
        assert tags["is_baseline"] == "false"


class TestInferModelArchitecture:
    def test_tft_detected_from_quantiles(self):
        cfg = {"model": {"quantiles": [0.25, 0.5, 0.75], "hidden_size": 64}}
        assert infer_model_architecture(cfg) == "tft"

    def test_lgbm_detected_from_n_estimators(self):
        cfg = {"model": {"n_estimators": 500, "learning_rate": 0.05}}
        assert infer_model_architecture(cfg) == "lgbm"

    def test_explicit_override(self):
        cfg = {"model_architecture": "nbeats"}
        assert infer_model_architecture(cfg) == "nbeats"


class TestSafeGitSha:
    def test_returns_str(self):
        sha = safe_git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0


class TestFileSha256:
    def test_hex_digest_length(self, tmp_path: Path):
        f = tmp_path / "x.txt"
        f.write_text("hello world")
        h = file_sha256(f)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_content_same_hash(self, tmp_path: Path):
        a = tmp_path / "a.txt"; a.write_text("content")
        b = tmp_path / "b.txt"; b.write_text("content")
        assert file_sha256(a) == file_sha256(b)

    def test_different_content_different_hash(self, tmp_path: Path):
        a = tmp_path / "a.txt"; a.write_text("a")
        b = tmp_path / "b.txt"; b.write_text("b")
        assert file_sha256(a) != file_sha256(b)


class TestComputeDatasetAudit:
    @pytest.fixture
    def synthetic_parquet(self, tmp_path: Path) -> Path:
        df = pd.DataFrame({
            "SC_CD": ["A", "A", "B", "B", "C", "C"],
            "STYLE_CD": ["S1", "S1", "S1", "S1", "S2", "S2"],
            "WEEK_START": pd.to_datetime(
                ["2025-01-06", "2025-01-13", "2025-01-06", "2025-01-13",
                 "2025-01-06", "2025-01-13"]
            ),
            "WEEKLY_SALE_QTY": [10, 0, 5, 7, 0, 0],
        })
        path = tmp_path / "tiny.parquet"
        df.to_parquet(path)
        return path

    def test_audit_keys_present(self, synthetic_parquet):
        df = pd.read_parquet(synthetic_parquet)
        audit = compute_dataset_audit(
            df, parquet_path=synthetic_parquet,
            cutoff="2025-01-13", val_end=None,
            group_key="SC_CD", style_col="STYLE_CD", target="WEEKLY_SALE_QTY",
        )
        for k in ("input_parquet_path", "input_parquet_sha256", "n_series_train",
                  "n_rows_train", "target_zero_ratio_train",
                  "style_list", "style_count", "n_series_per_style"):
            assert k in audit, f"missing key: {k}"

    def test_audit_counts(self, synthetic_parquet):
        df = pd.read_parquet(synthetic_parquet)
        audit = compute_dataset_audit(
            df, parquet_path=synthetic_parquet,
            cutoff="2025-01-13",
            group_key="SC_CD", style_col="STYLE_CD", target="WEEKLY_SALE_QTY",
        )
        assert audit["n_series_train"] == 3  # A, B, C
        assert audit["n_rows_train"] == 6
        assert audit["style_count"] == 2  # S1, S2
        # 3 zero rows out of 6
        assert audit["target_zero_ratio_train"] == pytest.approx(3 / 6)
        # style_list / n_series_per_style 가 json string
        assert json.loads(audit["style_list"]) == ["S1", "S2"]
        npstyle = json.loads(audit["n_series_per_style"])
        assert npstyle == {"S1": 2, "S2": 1}

    def test_val_end_optional(self, synthetic_parquet):
        df = pd.read_parquet(synthetic_parquet)
        audit_no_val = compute_dataset_audit(
            df, parquet_path=synthetic_parquet, cutoff="2025-01-06",
            group_key="SC_CD", target="WEEKLY_SALE_QTY",
        )
        assert "n_series_val_alive" not in audit_no_val

        audit_with_val = compute_dataset_audit(
            df, parquet_path=synthetic_parquet,
            cutoff="2025-01-06", val_end="2025-01-13",
            group_key="SC_CD", target="WEEKLY_SALE_QTY",
        )
        assert audit_with_val["n_series_val_alive"] == 3
        assert audit_with_val["n_rows_val"] == 3


class TestSystemEnvParams:
    def test_keys_present(self):
        env = system_env_params()
        assert "torch_version" in env
        assert "device" in env
        assert "pytorch_forecasting_version" in env
        assert env["device"] in {"mps", "cuda", "cpu", "unknown"}


# ── MLflow client mock tests ──
@pytest.fixture
def mock_mlflow_module(mocker):
    """mlflow 모듈의 핵심 함수를 mock 하고 mock 객체 dict 반환.

    `mlflow.start_run` 은 context manager 패턴이라 __enter__/__exit__ 도 처리.
    """
    mocker.patch("mlflow.set_tracking_uri")
    log_metric = mocker.patch("mlflow.log_metric")
    log_param = mocker.patch("mlflow.log_param")
    set_tag = mocker.patch("mlflow.set_tag")
    start_run = mocker.patch("mlflow.start_run")
    start_run.return_value.__enter__ = mocker.Mock(return_value=mocker.Mock(info=mocker.Mock(run_id="fake")))
    start_run.return_value.__exit__ = mocker.Mock(return_value=False)
    return {
        "log_metric": log_metric,
        "log_param": log_param,
        "set_tag": set_tag,
        "start_run": start_run,
    }


class TestMlflowMocks:
    def test_log_full_metrics_filters_nan_and_inf(self, mock_mlflow_module):
        from scripts.mlflow_logging import log_full_metrics

        metrics = {
            "valid.a": 0.5,
            "valid.b": 1.0,
            "nan_value": float("nan"),
            "inf_value": float("inf"),
            "string_value": "not_a_number",
        }
        n = log_full_metrics("fake_run_id", metrics, tracking_uri="http://fake")
        assert n == 2  # only valid.a and valid.b
        assert mock_mlflow_module["log_metric"].call_count == 2

    def test_log_params_safe_truncates_long(self, mock_mlflow_module):
        from scripts.mlflow_logging import log_params_safe

        long_val = "x" * 1000
        params = {"short": "abc", "long": long_val, "none": None, "list": [1, 2, 3]}
        n = log_params_safe("fake", params, tracking_uri="http://fake", max_value_len=100)
        assert n == 3  # short + long(truncated) + list  (none skip)
        calls = mock_mlflow_module["log_param"].call_args_list
        long_call = next(c for c in calls if c[0][0] == "long")
        assert len(long_call[0][1]) <= 100
        assert "[TRUNCATED]" in long_call[0][1]

    def test_set_run_tags_all_str(self, mock_mlflow_module):
        from scripts.mlflow_logging import set_run_tags
        n = set_run_tags("fake", {"a": "1", "b": 2}, tracking_uri="http://fake")
        assert n == 2
        for call in mock_mlflow_module["set_tag"].call_args_list:
            assert isinstance(call[0][1], str)


class TestAggregateMetricsToParent:
    def test_mean_median_std(self, mock_mlflow_module):
        from scripts.mlflow_logging import log_aggregate_metrics_to_parent

        children = [
            {"overall.wape_q50": 0.5, "horizon.wape_h1": 0.4},
            {"overall.wape_q50": 0.55, "horizon.wape_h1": 0.45},
            {"overall.wape_q50": 0.60, "horizon.wape_h1": 0.50},
        ]
        n = log_aggregate_metrics_to_parent(
            "fake_parent", children, tracking_uri="http://fake",
        )
        assert n == 6  # 2 keys × {mean, median, std}
        keys = [c[0][0] for c in mock_mlflow_module["log_metric"].call_args_list]
        assert "aggregate.mean.overall.wape_q50" in keys
        assert "aggregate.median.overall.wape_q50" in keys
        assert "aggregate.std.overall.wape_q50" in keys

    def test_empty_list_returns_zero(self, mock_mlflow_module):
        from scripts.mlflow_logging import log_aggregate_metrics_to_parent
        n = log_aggregate_metrics_to_parent("fake", [], tracking_uri="http://fake")
        assert n == 0
