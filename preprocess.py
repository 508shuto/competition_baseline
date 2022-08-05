from argparse import ArgumentParser

import pandas as pd

from .src.processing.create_kfold_df import CreateStrGrpKFold

parser = ArgumentParser()
parser.add_argument("--csv_path", action="store", type=str, default='./data/raw/input/train.csv')
parser.add_argument("--target_name", action="store", type=str, default='pm25_mid')
parser.add_argument("--output_path", action="store", type=str, default='./data/processed/preprocessed_train.csv')
args = parser.parse_args()

if __name__ == '__main__':
    csv_path = args.csv_path
    target_name = args.target_name
    output_path = args.output_path
    
    df = pd.read_csv(csv_path)
    creator = CreateStrGrpKFold(group='City')
    creator.create(df, target_name, output_path, regression=True)
