import time
from contextlib import contextmanager

import pandas as pd
from hydra.utils import get_original_cwd


@contextmanager
def timer(name):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def load_datasets(feats):
    dfs = [pd.read_feather(f"{get_original_cwd()}/data/processed/features/{f}_train.feather") for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f"{get_original_cwd()}/data/processed/features/{f}_test.feather") for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv(f"{get_original_cwd()}/data/raw/input/train.csv")
    y_train = train[target_name]
    return y_train
