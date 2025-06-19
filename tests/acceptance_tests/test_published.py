"""
Tests to check for agreement between outputs from source code and published results
"""
import csv
import os

import numpy as np

from usal_diaphragm.app import launch

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "_test_data", "Participants")


def test_P1():
    _run_on("P1")


def test_P2():
    _run_on("P2")


def test_P3():
    _run_on("P3")


def test_P4():
    _run_on("P4")


def test_P5():
    _run_on("P5")


def test_P6():
    _run_on("P6")


def test_P7():
    _run_on("P7")


def test_P8():
    _run_on("P8")


def test_P9():
    _run_on("P9")


def test_P10():
    _run_on("P10")


def test_P11():
    _run_on("P11")


def test_P12():
    _run_on("P12")


def _run_on(participant):
    os.chdir(os.path.join(DATA_ROOT, participant))

    if not os.path.exists(f"{participant}.yaml"):
        raise FileNotFoundError(
            f"Could not find {participant}.yaml in {os.getcwd()}.\n"
            "Download data from Figshare and move the Participants folder to _test_data."
        )

    launch.main([
        "--csv_suffix=-test",
        "--action=none",
        f"{participant}.yaml"
    ])


    published_filename = f"{participant}.vol_surface_params.csv"
    with open(published_filename, "rt") as fid:
        rdr = csv.reader(fid)
        published_params_table = [row for row in rdr]

    test_filename = f"{participant}.vol_surface_params-test.csv"
    with open(test_filename, "rt") as fid:
        rdr = csv.reader(fid)
        test_params_table = [row for row in rdr]

    assert published_params_table[0] == test_params_table[0]
    
    published_params = np.array([
        [float(v) for v in row]
        for row in published_params_table[1:]
    ])
    test_params = np.array([
        [float(v) for v in row]
        for row in test_params_table[1:]
    ])
    assert np.allclose(test_params, published_params)
