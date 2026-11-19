import sys

import pytest

from nemo_3_diarization.evaluate import main, parse_segments, score

REF = {"a": [(0.0, 5.0)], "b": [(5.0, 10.0)]}


def test_score_perfect_and_permuted():
    assert score(REF, REF)["der"] == 0.0
    assert score(REF, {"x": [(0.0, 5.0)], "y": [(5.0, 10.0)]})["der"] == 0.0


def test_score_collar_forgives_quarter_second():
    assert score(REF, {"x": [(0.0, 5.2)], "y": [(5.2, 10.0)]})["der"] == 0.0


def test_score_empty_hypothesis():
    s = score(REF, {})
    assert s["der"] == 1.0 and s["miss"] > 0


def test_parse_segments():
    assert parse_segments(["0.00 1.50 speaker_0", "2.00 3.00 speaker_1"]) == {
        "speaker_0": [(0.0, 1.5)], "speaker_1": [(2.0, 3.0)]}


def test_score_unsorted_overlapping_reference():
    ref = {"a": [(2.0, 5.0), (0.0, 4.0)], "b": [(5.0, 10.0)]}
    s = score(ref, {"x": [(0.0, 5.0)], "y": [(5.0, 10.0)]})
    assert s["der"] == 0.0 and s["total"] == score(REF, REF)["total"]


def test_empty_manifest_exits_before_model_load(tmp_path, monkeypatch, capsys):
    (tmp_path / "empty.json").write_text("\n")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(sys, "argv", ["x", "--models", "m", "--manifests", str(tmp_path / "empty.json"),
                                      "--out", str(tmp_path / "r.json")])
    with pytest.raises(SystemExit):
        main()
    assert "empty manifest(s)" in capsys.readouterr().err and not (tmp_path / "r.json").exists()


def test_missing_mlflow_uri_exits_before_model_load(tmp_path, monkeypatch, capsys):
    (tmp_path / "m.json").write_text('{"audio_filepath": "a.wav"}\n')
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(sys, "argv", ["x", "--models", "m", "--manifests", str(tmp_path / "m.json"),
                                      "--out", str(tmp_path / "r.json")])
    with pytest.raises(SystemExit):
        main()
    assert "MLFLOW_TRACKING_URI is not set" in capsys.readouterr().err
