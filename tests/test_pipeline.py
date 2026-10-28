import json
import os

import pytest

from pid import cli

REQUIRED_TOP_LEVEL = {
    "schema_version", "data_source", "generated_at", "git_commit", "seed", "datasets", "cleaning",
    "threshold_policy", "cross_dataset", "models", "artifacts", "notes",
}


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("run")
    old = os.getcwd()
    os.chdir(base)  # outputs/ vai para a pasta temporária
    try:
        out = base / "reports"
        metrics = cli.run(None, synthetic=True, out_dir=str(out))
    finally:
        os.chdir(old)
    return out, metrics


def test_synthetic_pipeline_writes_contract_and_figures(pipeline_run):
    out, metrics = pipeline_run

    written = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert REQUIRED_TOP_LEVEL <= set(written)
    assert written["schema_version"] == 1
    assert written["data_source"] == "synthetic"
    assert any("SINTÉTICOS" in n for n in written["notes"])
    assert {m["name"] for m in written["models"]} == {"rules", "tfidf_lr"}

    tfidf = next(m for m in written["models"] if m["name"] == "tfidf_lr")
    for key in ("in_distribution", "external", "latency_ms_cpu", "in_distribution_ci95"):
        assert key in tfidf
    lo, hi = tfidf["in_distribution_ci95"]["f1"]
    assert lo <= tfidf["in_distribution"]["f1"] <= hi or lo == hi

    for rel in written["artifacts"].values():
        assert (out / rel).exists()
    assert metrics["cross_dataset"] == {"train_on": "synthetic_a", "test_on": "synthetic_b"}


def test_cleaning_report_is_present(pipeline_run):
    _, metrics = pipeline_run
    assert metrics["cleaning"]["removed_from_primary_for_overlap"] >= 0
    assert "duplicates_removed" in metrics["cleaning"]["primary"]


def test_error_files_are_written_for_manual_reading(pipeline_run):
    out, _ = pipeline_run
    for name in ("errors_tfidf_in_distribution.csv", "errors_tfidf_external.csv"):
        header = (out / name).read_text(encoding="utf-8").splitlines()[0]
        assert header == "set,type,label,score,text"


def test_extra_baselines_adds_models_and_benchmark_charts(tmp_path):
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        out = tmp_path / "reports_extra"
        metrics = cli.run(None, synthetic=True, out_dir=str(out), extra_baselines=True)
    finally:
        os.chdir(old)
    names = {m["name"] for m in metrics["models"]}
    assert {"rules", "tfidf_lr", "char_wb_tfidf_lr", "word_char_svm"} <= names
    for art_key in ("benchmark_f1", "benchmark_roc_auc", "benchmark_latency_pareto"):
        assert art_key in metrics["artifacts"]
        assert (out / metrics["artifacts"][art_key]).exists()
