from argparse import ArgumentParser

import joblib
import numpy as np
import pandas as pd

from src.modeling.model import LGBMRegressor, XGBRegressor
from src.preparation.logger import get_logger
from src.processing.runner import Runner
import settings

# logger = get_logger()

parser = ArgumentParser()
# parser.add_argument("--experiment_name", action="store", type=str, default='./data/raw/input/train.csv')
# parser.add_argument("--train_dataset", action="store", type=str, default='le_train.csv')
# parser.add_argument("--output_path", action="store", type=str, default='./data/processed/preprocessed_train.csv')
# args = parser.parse_args()

def main():
    # experiment_name = args.experiment_name
    # train_dataset = args.train_dataset
    # output_path = args.output_path
    
    train_df = pd.read_csv(settings.TRAIN_CSV_PATH)
    test_df = pd.read_csv(settings.TEST_CSV_PATH)
    
    target = settings.TARGET
    train_X = train_df.drop(target, axis=1)
    train_y = train_df.loc[:, [target]]
    
    lgbm_params = joblib.load(settings.LGBM_PARAM_PATH)
    xgb_params = joblib.load(settings.XGB_PARAM_PATH)

    # lgbによる学習・予測
    runner = Runner('lgb', LGBMRegressor, lgbm_params)
    runner.train_cv(train_X, train_y)
    runner.predict_cv(test_df)
    # Submission.create_submission('lgb')

    # xgboostによる学習・予測
    # runner = Runner('xgb', XGBRegressor, xgb_params)
    # runner.train_cv(train_X, train_y)
    # runner.predict_cv(test_df)
    # Submission.create_submission('xgb')

if __name__ == '__main__':
    main()
