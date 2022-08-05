import numpy as np
import pandas as pd
from sklearn.model_selection import (KFold, StratifiedGroupKFold,
                                     StratifiedKFold)


class CreateFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=1234, group=None):
        self.kfold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
            )
        self.group = group
        
    def _split(self, X, y=None):
        group = X[self.group]
        return self.kfold.split(X=X, y=y)
    
    def create(self, df, target, path, regression=False):
        df['kfold'] = -1
        df = df.sample(frac=1).reset_index(drop=True)

        if regression:
            num_bins = int(np.floor(1 + np.log2(len(df))))
            df.loc[:, 'bins'] = pd.cut(
                df[target], bins=num_bins, labels=False
            )
            y = df.bins
        else:
            y = target
        
        for f, (t_, v_) in enumerate(self._split(X=df, y=y)):
            df.loc[v_, 'kfold'] = f

        if regression:
            df = df.drop('bins', axis=1).sort_values(by='id').reset_index(drop=True)
        df.to_csv(path, index=False)

class CreateStrGrpKFold(CreateFold):
    def __init__(self, group, n_splits=5, shuffle=True, random_state=1234):
        self.kfold = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
            )
        self.group = group
    
    def _split(self, X, y=None):
        group = X[self.group]
        return self.kfold.split(X=X, y=y, groups=group)
