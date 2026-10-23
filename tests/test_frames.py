import numpy as np

from envtok_mace.data.frames import Frames, _ranges


def test_ranges_matches_python_loop():
    starts = np.array([0, 5, 9, 9])
    stops = np.array([3, 7, 9, 12])
    expected = np.concatenate([np.arange(s, e) for s, e in zip(starts, stops)])
    np.testing.assert_array_equal(_ranges(starts, stops), expected)


def test_save_load_roundtrip(val40, tmp_path):
    path = tmp_path / "f.h5"
    val40.save(path)
    back = Frames.load(path)
    assert back.sid.tolist() == val40.sid.tolist()
    for name in ("atom_ptr", "numbers", "positions", "forces", "cell", "pbc", "energy", "stress",
                 "edge_ptr", "edge_index", "unit_shifts"):
        np.testing.assert_array_equal(getattr(back, name), getattr(val40, name))


def test_subset_then_concatenate_restores_order(val40):
    idx = np.arange(len(val40))
    parts = [val40.subset(idx[:13]), val40.subset(idx[13:27]), val40.subset(idx[27:])]
    whole = Frames.concatenate(parts)
    np.testing.assert_array_equal(whole.positions, val40.positions)
    np.testing.assert_array_equal(whole.edge_index, val40.edge_index)
    np.testing.assert_array_equal(whole.edge_ptr, val40.edge_ptr)


def test_subset_picks_the_right_frames(val40):
    idx = np.array([7, 2, 31])
    sub = val40.subset(idx)
    for k, i in enumerate(idx):
        a, b = sub.get(k), val40.get(int(i))
        assert a["sid"] == b["sid"]
        np.testing.assert_array_equal(a["positions"], b["positions"])
        np.testing.assert_array_equal(a["edge_index"], b["edge_index"])
