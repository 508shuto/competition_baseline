
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from ..preparation.logger import get_logger

logger = get_logger()


class Runner:

    def __init__(self, run_name: str, model_cls, params: dict):
        """コンストラクタ

        :param run_name: ランの名前
        :param model_cls: モデルのクラス
        :param features: 特徴量のリスト
        :param params: ハイパーパラメータ
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.params = params
        self.n_fold = 5

    def train_cv(self, train_X: pd.DataFrame, train_y):
        models = []
        losses = []
        
        folds = len(train_X['kfold'].unique())
        for fold in range(folds):
            logger.info(f'================fold:{fold:02d} start================')
            model, loss = self.train_fold(train_X, train_y, fold)
            models.append(model.model)
            losses.append(loss)
            
        losses_mean = np.mean(losses,axis=0)
        logger.info(f'loss mean: {losses_mean}')
        #return models, losses_mean

    def predict_cv(self, test_X: pd.DataFrame):
        preds = []
        for fold in range(self.n_fold):
            model = joblib.load(f'./model/{self.run_name}_{fold:02d}.pkl')
            y_pred = model.predict(test_X, model)
            preds.append(y_pred)
            
        preds_array = np.array(preds)
        preds_mean = np.mean(preds_array,axis=0)
        return preds_mean

    def train_fold(self, train_X, train_y, fold):
        X_train, y_train, X_valid, y_valid = self.split_train_val(train_X, train_y, fold)
        model = self.model_cls(params=self.params)
        model.train(X_train, y_train, X_valid, y_valid)
        model.plot_result()
        
        os.makedirs('./model', exist_ok=True)
        model.save_model(f'./model/{self.run_name}_{fold:02d}.pkl')
        
        y_pred = model.predict(X_valid)
        loss = np.sqrt(mean_squared_error(y_valid, y_pred))
        logger.info(f'fold {fold:02d} loss: {loss}')
        return model, loss

    def split_train_val(self, X, y, fold):
        X_train = X.loc[X['kfold']!=fold, :]
        y_train = y.loc[X['kfold']!=fold, :]
        X_valid = X.loc[X['kfold']==fold, :]
        y_valid = y.loc[X['kfold']==fold, :]
        return X_train, y_train, X_valid, y_valid
    
    @classmethod
    def create_submission(cls, run_name):
        submission = pd.read_csv('./input/sampleSubmission.csv')
        pred = joblib.load(f'./model/{run_name}-test.pkl')
        for i in range(pred.shape[1]):
            submission[f'Class_{i + 1}'] = pred[:, i]
        submission.to_csv(f'./submission/{run_name}.csv', index=False)
