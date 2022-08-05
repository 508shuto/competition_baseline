from sklearn.model_selection import KFold as KF


class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=1234):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kf = KF(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
    
    def split(self, X, y=None):
        return self.kf.split(X)
    
class GroupKFold(KFold):
    def __init__(self, group, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group = group
        
    def split(self, X=None, y=None):
        return self.kf.split(self.group)

if __name__ == '__main__':
    kf = GroupKFold(group='City', n_splits=5, shuffle=True, random_state=71)
    print(kf.__class__ == GroupKFold)
    print(kf.random_state)
    print(kf.group)

