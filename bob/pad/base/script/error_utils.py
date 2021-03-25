#!/usr/bin/env python
# Ivana Chingovska <ivana.chingovska@idiap.ch>
# Fri Dec  7 12:33:37 CET 2012
"""Utility functions for computation of EPSC curve and related measurement"""

from bob.measure import far_threshold, eer_threshold, min_hter_threshold, farfrr, frr_threshold
from bob.bio.base.score.load import four_column
from collections import defaultdict
import re


def calc_threshold(method, pos, negs, all_negs, far_value=None, is_sorted=False):
    """Calculates the threshold based on the given method.

    Parameters
    ----------
    method : str
        One of ``bpcer20``, ``eer``, ``min-hter``, ``apcer20``.
    pos : array_like
        The positive scores. They should be sorted!
    negs : list
        A list of array_like negative scores. Each item in the list corresponds to
        scores of one PAI.
    all_negs : array_like
        An array of all negative scores. This can be calculated from negs as well but we
        ask for it since you might have it already calculated.
    far_value : None, optional
        If method is far, far_value and all_negs are used to calculate the threshold.
    is_sorted : bool, optional
        If True, it means all scores are sorted and no sorting will happen.

    Returns
    -------
    float
        The calculated threshold.

    Raises
    ------
    ValueError
        If method is unknown.
    """
    method = method.lower()
    if "bpcer" in method:
        desired_apcer = 1 / float(method.replace("bpcer", ""))
        threshold = apcer_threshold(desired_apcer, pos, *negs, is_sorted=is_sorted)
    elif "apcer" in method:
        desired_bpcer = 1 / float(method.replace("apcer", ""))
        threshold = frr_threshold(all_negs, pos, desired_bpcer, is_sorted=is_sorted)
    elif method == "far":
        threshold = far_threshold(all_negs, pos, far_value, is_sorted=is_sorted)
    elif method == "eer":
        threshold = eer_threshold(all_negs, pos, is_sorted=is_sorted)
    elif method == "min-hter":
        threshold = min_hter_threshold(all_negs, pos, is_sorted=is_sorted)
    else:
        raise ValueError("Unknown threshold criteria: {}".format(method))

    return threshold


def apcer_threshold(desired_apcer, pos, *negs, is_sorted=False):
    """Computes the threshold given the desired APCER as the criteria.

    APCER is computed as max of all APCER_PAI values.
    The threshold will be computed such that the real APCER is **at most** the desired
    value.

    Parameters
    ----------
    desired_apcer : float
        The desired APCER value.
    pos : list
        An array or list of positive scores in float.
    *negs
        A list of negative scores. Each item corresponds to the negative scores of one
        PAI.
    is_sorted : bool, optional
        Set to ``True`` if ALL arrays (pos and negs) are sorted.

    Returns
    -------
    float
        The computed threshold that satisfies the desired APCER.
    """
    threshold = max(
        far_threshold(neg, pos, desired_apcer, is_sorted=is_sorted) for neg in negs
    )
    return threshold


def apcer_bpcer(threshold, pos, *negs):
    """Computes APCER_PAI, APCER, and BPCER given the positive scores and a list of
    negative scores and a threshold.

    Parameters
    ----------
    threshold : float
        The threshold to be used to compute the error rates.
    pos : list
        An array or list of positive scores in float.
    *negs
        A list of negative scores. Each item corresponds to the negative scores of one
        PAI.

    Returns
    -------
    tuple
        A tuple such as (list of APCER_PAI, APCER, BPCER)
    """
    apcers = []
    assert len(negs) > 0, negs
    for neg in negs:
        far, frr = farfrr(neg, pos, threshold)
        apcers.append(far)
    bpcer = frr  # bpcer will be the same in all cases
    return apcers, max(apcers), bpcer


def negatives_per_pai_and_positives(filename, regexps=None, regexp_column="real_id"):
    """Returns scores for Bona-Fide samples and scores for each PAI.
    By default, the real_id column (second column) is used as indication for each
    Presentation Attack Instrument (PAI).

    For example, if you have scores like:
        001 001    bona_fide_sample_1_path 0.9
        001 print  print_sample_1_path     0.6
        001 print  print_sample_2_path     0.6
        001 replay replay_sample_1_path    0.2
        001 replay replay_sample_2_path    0.2
        001 mask   mask_sample_1_path      0.5
        001 mask   mask_sample_2_path      0.5
    this function will return 3 sets of negative scores (for each print, replay, and
    mask PAIs).

    Otherwise, you can provide a list regular expressions that match each PAI.
    For example, if you have scores like:
        001 001      bona_fide_sample_1_path 0.9
        001 print/1  print_sample_1_path     0.6
        001 print/2  print_sample_2_path     0.6
        001 replay/1 replay_sample_1_path    0.2
        001 replay/2 replay_sample_2_path    0.2
        001 mask/1   mask_sample_1_path      0.5
        001 mask/2   mask_sample_2_path      0.5
    and give a list of regexps as ('print', 'replay', 'mask') the function will return 3
    sets of negative scores (for each print, replay, and mask PAIs).


    Parameters
    ----------
    filename : str
        Path to the score file.
    regexps : None, optional
        A list of regular expressions that match each PAI. If not given, the values in
        the real_id column are used to find scores for different PAIs.
    regexp_column : str, optional
        If a list of regular expressions are given, those patterns will be matched
        against the values in this column.

    Returns
    -------
    tuple
        A tuple containing pos scores and a dict of negative scores mapping PAIs to
        their scores.

    Raises
    ------
    ValueError
        If none of the given regular expressions match the values in regexp_column.
    """
    pos = []
    negs = defaultdict(list)
    if regexps:
        regexps = [re.compile(pattern) for pattern in regexps]
        assert regexp_column in ("claimed_id", "real_id", "test_label"), regexp_column

    for claimed_id, real_id, test_label, score in four_column(filename):
        # if it is a Bona-Fide score
        if claimed_id == real_id:
            pos.append(score)
            continue
        if not regexps:
            negs[real_id].append(score)
            continue
        # if regexps is not None or empty and is not a Bona-Fide score
        string = {
            "claimed_id": claimed_id,
            "real_id": real_id,
            "test_label": test_label,
        }[regexp_column]
        for pattern in regexps:
            if pattern.match(string):
                negs[pattern.pattern].append(score)
                break
        else:  # this else is for the for loop: ``for pattern in regexps:``
            raise ValueError(
                f"No regexps: {regexps} match `{string}' from `{regexp_column}' column"
            )
    return pos, negs
