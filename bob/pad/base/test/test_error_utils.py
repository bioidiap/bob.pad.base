import h5py
import nose
import numpy as np

from bob.io.base.test_utils import datafile
from bob.pad.base.error_utils import (
    apcer_bpcer,
    calc_threshold,
    split_csv_pad,
    split_csv_pad_per_pai,
)

GENERATE_REFERENCES = False

scores_dev = datafile("per_pai_scores/scores-dev.csv", module=__name__)
scores_dev_reference_mask = datafile(
    "per_pai_scores/scores-dev-{i}.hdf5", module=__name__
)


def _dump_dict(f, d, name):
    f[f"{name}_len"] = len(d)
    for i, (k, v) in enumerate(d.items()):
        f[f"{name}_key_{i}"] = k
        f[f"{name}_value_{i}"] = v


def _read_dict(f, name):
    ret = dict()
    for i in range(int(np.array(f[f"{name}_len"]))):
        k = np.array(f[f"{name}_key_{i}"])[0].decode()
        v = np.array(f[f"{name}_value_{i}"])
        if v.size > 1:
            v = v.tolist()
        else:
            v = v[0]
            if isinstance(v, bytes):
                v = v.decode()
        ret[k] = v
    return ret


def test_per_pai_apcer():
    for i, regexps in enumerate(
        (None, ["x[0-2]", "x[3-4]"], ["x[1-2]", "x[3-4]"])
    ):
        try:
            pos, negs = split_csv_pad_per_pai(scores_dev, regexps)
        except ValueError:
            if i == 2:
                continue
            raise
        all_negs = [s for scores in negs.values() for s in scores]

        thresholds = dict()
        for method in ("bpcer20", "far", "eer", "min-hter"):
            thresholds[method] = calc_threshold(
                method, pos, negs.values(), all_negs, far_value=0.1
            )

        metrics = dict()
        for method, threshold in thresholds.items():
            apcers, apcer, bpcer = apcer_bpcer(threshold, pos, *negs.values())
            metrics[method] = apcers + [apcer, bpcer]

        scores_dev_reference = scores_dev_reference_mask.format(i=i)
        if GENERATE_REFERENCES:
            with h5py.File(scores_dev_reference, "w") as f:
                f["pos"] = pos
                _dump_dict(f, negs, "negs")
                _dump_dict(f, thresholds, "thresholds")
                _dump_dict(f, metrics, "metrics")

        with h5py.File(scores_dev_reference, "r") as f:
            ref_pos = np.array(f["pos"]).tolist()
            ref_negs = _read_dict(f, "negs")
            ref_thresholds = _read_dict(f, "thresholds")
            ref_metrics = _read_dict(f, "metrics")

        nose.tools.assert_list_equal(pos, ref_pos)
        nose.tools.assert_dict_equal(negs, ref_negs)
        nose.tools.assert_dict_equal(thresholds, ref_thresholds)
        nose.tools.assert_dict_equal(metrics, ref_metrics)


def test_csv_split():
    neg, pos = split_csv_pad(scores_dev)
    assert len(neg) == 5000, len(neg)
    assert len(pos) == 5000, len(pos)
    assert np.isclose(np.mean(neg), -10, atol=0.1), np.mean(neg)
    assert np.isclose(np.mean(pos), 10, atol=0.1), np.mean(pos)
