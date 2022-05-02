#!/usr/bin/env python
# Ivana Chingovska <ivana.chingovska@idiap.ch>
# Fri Dec  7 12:33:37 CET 2012
"""Utility functions for computation of EPSC curve and related measurement"""

import logging
import re

from collections import defaultdict

import numpy

from bob.bio.base.score.load import _iterate_csv_score_file
from bob.measure import (
    eer_threshold,
    far_threshold,
    farfrr,
    frr_threshold,
    min_hter_threshold,
)

logger = logging.getLogger(__name__)


def calc_threshold(
    method, pos, negs, all_negs, far_value=None, is_sorted=False
):
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
        threshold = apcer_threshold(
            desired_apcer, pos, *negs, is_sorted=is_sorted
        )
    elif "apcer" in method:
        desired_bpcer = 1 / float(method.replace("apcer", ""))
        threshold = frr_threshold(
            all_negs, pos, desired_bpcer, is_sorted=is_sorted
        )
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
        far_threshold(neg, pos, desired_apcer, is_sorted=is_sorted)
        for neg in negs
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


def split_csv_pad_per_pai(filename, regexps=[], regexp_column="attack_type"):
    """Returns scores for Bona-Fide samples and scores for each PAI.
    By default, the real_id column (second column) is used as indication for each
    Presentation Attack Instrument (PAI).

    For example, with default regexps and regexp_column, if you have scores like:
        claimed_id, test_label,              is_bonafide, attack_type, score
        001,        bona_fide_sample_1_path, True,        ,            0.9
        001,        print_sample_1_path,     False,       print,       0.6
        001,        print_sample_2_path,     False,       print,       0.6
        001,        replay_sample_1_path,    False,       replay,      0.2
        001,        replay_sample_2_path,    False,       replay,      0.2
        001,        mask_sample_1_path,      False,       mask,        0.5
        001,        mask_sample_2_path,      False,       mask,        0.5
    this function will return 1 set of positive scores, and 3 sets of negative scores
    (for each print, replay, and mask PAIs).

    Otherwise, you can provide a list regular expressions that match each PAI.
    For example, with regexps as ['print', 'replay', 'mask'], if you have scores like:
        claimed_id, test_label,              is_bonafide, attack_type, score
        001,        bona_fide_sample_1_path, True,        ,            0.9
        001,        print_sample_1_path,     False,       print/1,     0.6
        001,        print_sample_2_path,     False,       print/2,     0.6
        001,        replay_sample_1_path,    False,       replay/1,    0.2
        001,        replay_sample_2_path,    False,       replay/2,    0.2
        001,        mask_sample_1_path,      False,       mask/1,      0.5
        001,        mask_sample_2_path,      False,       mask/2,      0.5
    the function will return 3 sets of negative scores (for print, replay, and mask
    PAIs, given in regexp).


    Parameters
    ----------
    filename : str
        Path to the score file.
    regexps : List of str, optional
        A list of regular expressions that match each PAI. If not given, the values in
        the column pointed by regexp_column are used to find scores for different PAIs.
    regexp_column : str, optional
        If a list of regular expressions are given, those patterns will be matched
        against the values in this column. default: ``attack_type``

    Returns
    -------
    tuple ([positives], {'pai_name': [negatives]})
        A tuple containing positive scores and a dict of negative scores mapping PAIs
        names to their respective scores.

    Raises
    ------
    ValueError
        If none of the given regular expressions match the values in regexp_column.
    KeyError
        If regexp_column is not a column of the CSV file.
    """
    pos = []
    negs = defaultdict(list)
    logger.debug(f"Loading CSV score file: '{filename}'")
    if regexps:
        regexps = [re.compile(pattern) for pattern in regexps]

    for row in _iterate_csv_score_file(filename):
        # if it is a Bona-Fide score
        if row["is_bonafide"].lower() == "true":
            pos.append(row["score"])
            continue
        if not regexps:
            negs[row[regexp_column]].append(row["score"])
            continue
        # if regexps is not None or empty and is not a Bona-Fide score
        for pattern in regexps:
            if pattern.search(row[regexp_column]):
                negs[pattern.pattern].append(row["score"])
                break
        else:  # this else is for the for loop: ``for pattern in regexps:``
            raise ValueError(
                f"No regexps: {regexps} match `{row[regexp_column]}' "
                f"from `{regexp_column}' column."
            )
    logger.debug(f"Found {len(negs)} different PAIs names: {list(negs.keys())}")
    return pos, negs


def split_csv_pad(filename):
    """Loads PAD scores from a CSV score file, splits them by attack vs bonafide.

    The CSV must contain a ``is_bonafide`` column with each field either
    ``True`` or ``False`` (case insensitive).

    Parameters
    ----------
    filename: str
        The path to a CSV file containing all the scores.

    Returns
    -------
    (attack, bonafide): Tuple of 1D-arrays
        The negative (attacks) and positives (bonafide) scores.
    """
    logger.debug(f"Loading CSV score file: '{filename}'")
    split_scores = defaultdict(list)
    for row in _iterate_csv_score_file(filename):
        if row["is_bonafide"].lower() == "true":
            split_scores["bonafide"].append(row["score"])
        else:
            split_scores["attack"].append(row["score"])
    logger.debug(
        f"Found {len(split_scores['attack'])} negative (attack), and"
        f"{len(split_scores['bonafide'])} positive (bonafide) scores."
    )
    # Cast the scores to numpy float
    for key, scores in split_scores.items():
        split_scores[key] = numpy.array(scores, dtype=numpy.float64)
    return split_scores["attack"], split_scores["bonafide"]
