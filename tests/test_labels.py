from nemo_3_diarization.labels import *

def test_merge_pads_and_joins():
    assert merge([(1.0, 2.0), (2.5, 3.0)], pad=0.3) == [(0.7, 3.3)]
    assert merge([(0.1, 0.2)], pad=0.3) == [(0.0, 0.5)]

def test_intersect():
    assert intersect([(0, 5), (8, 10)], [(4, 9)]) == [(4, 5), (8, 9)]

def test_windows_drop_muted():
    assert windows(230.0, muted=[], win=105.0) == [(0.0, 105.0), (105.0, 210.0)]
    assert windows(230.0, muted=[(104.99, 105.0)], win=105.0) == [(105.0, 210.0)]

def test_top_speakers():
    act = {"a": [(0, 10)], "b": [(10, 30)], "c": [(30, 33)]}
    assert top_speakers(act, 0, 105, n=2, min_speech=5) == ["b", "a"]
    assert top_speakers(act, 0, 105, n=8, min_speech=0.5) == ["b", "a", "c"]

def test_rttm_roundtrip(tmp_path):
    lines = rttm_lines("w1", {"a": [(100, 110)], "b": [(200, 220)]}, 105, 210, ["a", "b"])
    assert lines[0] == "SPEAKER w1 1 0.000 5.000 <NA> <NA> a <NA> <NA>"
    (tmp_path / "w1.rttm").write_text("\n".join(lines) + "\n")
    assert read_rttm(tmp_path / "w1.rttm") == {"a": [(0.0, 5.0)], "b": [(95.0, 105.0)]}

def test_split_is_deterministic_and_proportional():
    s = [split_of(f"conv{i}") for i in range(10000)]
    assert s == [split_of(f"conv{i}") for i in range(10000)]
    assert 0.87 < s.count("train") / 10000 < 0.93
