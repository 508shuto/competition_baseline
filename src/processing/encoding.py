import copy
from argparse import ArgumentParser

import pandas as pd
from sklearn.preprocessing import LabelEncoder

parser = ArgumentParser()
parser.add_argument("train_csv", action="store", type=str)
parser.add_argument("--test_csv", action="store", type=str, default=None)

parser.add_argument("--target_name", action="store", type=str, default='pm25_mid')
parser.add_argument("--output_path", action="store", type=str, default='./data/processed/te_train.csv')
args = parser.parse_args()

class TagetEncoder:
    def __init__(self, target_col, rm_cols=None, num_cols=None) -> None:
        self.target_col = target_col
        self.rm_cols = rm_cols
        self.num_cols = num_cols
        self.features = None
    
    def mean_target_encoding(self, data: pd.DataFrame, train_data=None) -> pd.DataFrame:

        df = copy.deepcopy(data)
        df = self._preprocess(df)

        if train_data:
            df_train = copy.deepcopy(train_data)
            df_train = self._preprocess(df_train)
            encoded_df = self._mapping(df_train, df)
            encoded_df.to_csv(args.output_path.replace('train', 'test'), index=False)
        else:
            encoded_dfs = []
            for fold in range(df.kfold.max()):
                df_train = df[df.kfold != fold].reset_index(drop=True)
                df_valid = df[df.kfold == fold].reset_index(drop=True)
                df_valid = self._mapping(df_train, df_valid)
                encoded_dfs.append(df_valid)
            encoded_df = pd.concat(encoded_dfs, axis=0)
            encoded_df.to_csv(args.output_path, index=False)
        return encoded_df
    
    def _preprocess(self, df: pd.DataFrame):
        self.features = [
                    f for f in df.columns if f not in (self.target_col, self.rm_cols)
                    and f not in self.num_cols
                    ]

        for col in self.features:
            if col not in self.num_cols:
                df.loc[:, col] = df[col].astype(str).fillna('NONE')
                lbl = LabelEncoder()
                lbl.fit(df[col])
                df.loc[:, col] = lbl.transform(df[col])
        return df

    def _mapping(self, base_df, mapping_df):
        if self.features is None:
            print("You don't run preprocess!!")
        else:
            for column in self.features:
                mapping_dict = dict(
                    base_df.groupby(column)[self.target_col].mean()
                )
                mapping_df.loc[:, column + '_enc'] = mapping_df[column].map(mapping_dict)
        return mapping_df
    
if __name__ == '__main__':
    train_csv_path = args.train_csv
    test_csv_path = args.test_csv
    target_name = args.target_name
    # output_path = args.output_path
    
    num_cols = [
        '地域',
        '市区町村コード',
        '最寄駅：距離（分）',
        '面積（㎡）',   
        '建築年',
        '建ぺい率（％）',
        '容積率（％）',
        '取引時点',
    ]

    te = TagetEncoder(
        target_col=target_name,
        rm_cols=['ID', 'kfold'],
        num_cols=num_cols,
        )
    
    train_df = pd.read_csv(train_csv_path)
    train_df = te.mean_target_encoding(train_df)
    if test_csv_path:
        test_df = pd.read_csv(test_csv_path)
        test_df = te.mean_target_encoding(test_df, train_df)    
