import yaml
from typing import List

import pandas as pd


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def concat_datasets(data_paths: List[str], *args, **kwargs):
    dfs = [pd.read_csv(path, index_col=0, *args, **kwargs) for path in data_paths]
    return pd.concat(dfs)


def print_verbose(what: str, verbose: bool, *args, **kwargs):
    if verbose:
        print(what, *args, **kwargs)
