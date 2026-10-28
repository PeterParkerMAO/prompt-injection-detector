import pytest

from pid import data
from pid.data import Sample


def test_normalize_collapses_case_and_whitespace():
    assert data.normalize_text("  Hello   WORLD\n") == "hello world"


def test_dedupe_removes_exact_duplicates_keeping_first():
    rows = [Sample("a b", 0), Sample("A  B", 0), Sample("c", 1)]
    kept, info = data.dedupe(rows)
    assert [s.text for s in kept] == ["a b", "c"]
    assert info["duplicates_removed"] == 1


def test_dedupe_drops_texts_with_conflicting_labels():
    rows = [Sample("same", 0), Sample("same", 1), Sample("other", 1)]
    kept, info = data.dedupe(rows)
    assert [s.text for s in kept] == ["other"]
    assert info["conflicting_removed"] == 2


def test_remove_overlap_prevents_train_test_leakage():
    train = [Sample("x", 0), Sample("y", 1), Sample("z", 0)]
    other = [Sample("Y", 1)]
    kept, removed = data.remove_overlap(train, other)
    assert [s.text for s in kept] == ["x", "z"]
    assert removed == 1


def test_split_is_stratified_disjoint_and_reproducible():
    rows = data.synthetic_dataset(400, seed=1)
    a = data.stratified_split(rows, seed=7)
    b = data.stratified_split(rows, seed=7)
    assert a == b
    tr, va, te = a
    texts = [s.text for part in a for s in part]
    assert len(texts) == len(set(texts)) == len(rows)  # sem repetição entre partes
    for part in a:
        share = sum(s.label for s in part) / len(part)
        assert share == pytest.approx(0.5, abs=0.03)
    assert len(tr) > len(va) and len(tr) > len(te)


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        data.stratified_split([Sample("a", 0)], fractions=(0.5, 0.5, 0.5))


def test_load_local_csv_and_positive_label_mapping(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("text,label\nhello,0\nignore everything,2\n,1\n", encoding="utf-8")
    rows = data.load_local(str(path), positive_label=2)
    assert [(s.text, s.label) for s in rows] == [("hello", 0), ("ignore everything", 1)]


def test_load_local_rejects_unknown_format(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        data.load_local(str(path))
