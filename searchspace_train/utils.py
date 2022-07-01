from typing import List

import pandas as pd


def load_config(config_path: str):
    pass


def concat_datasets(data_paths: List[str], *args, **kwargs):
    dfs = [pd.DataFrame(path, *args, **kwargs) for path in data_paths]
    return pd.concat(dfs)


def print_verbose(what, verbose, *args, **kwargs):
    if verbose:
        print(what, *args, **kwargs)
