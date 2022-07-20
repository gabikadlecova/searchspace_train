import os
import sys

import pandas as pd
from click.testing import CliRunner
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '..', '..', '..', 'scripts'))
from concat_datasets import run


def run_on_three(first, second, out, ref):
    runner = CliRunner()
    runner.invoke(run, [first, second, out])

    assert os.path.exists(out)
    ref_df = pd.read_csv(ref, index_col=0)
    out_df = pd.read_csv(out, index_col=0)
    assert ref_df.equals(out_df)

    os.remove(out)


def test_run_concat(data_dir):
    first = os.path.join(data_dir, 'nb_1.csv')
    second = os.path.join(data_dir, 'nb_2.csv')
    res = os.path.join(data_dir, 'nb_res.csv')
    out = os.path.join(data_dir, 'nb_out.csv')

    run_on_three(first, second, out, res)


def test_run_drop_duplicates(data_dir):
    same = os.path.join(data_dir, 'saved_dataset.csv')
    out = os.path.join(data_dir, 'nb_out.csv')

    run_on_three(same, same, out, same)
