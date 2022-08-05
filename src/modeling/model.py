from typing import Optional
from abc import ABC, abstractmethod

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb


class Model(ABC):
    
    def __init__(self, params: dict) -> None:
        """コンストラクタ
        
        :param params: ハイパーパラメータ
        """
        self.params = params
        self.model = None

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_valid: Optional[pd.DataFrame] = None,
              y_valid: Optional[pd.Series] = None) -> None:
        """モデルの学習を行い、学習済のモデルを保存する

        :param X_train: 学習データの特徴量
        :param y_train: 学習データの目的変数
        :param X_valid: バリデーションデータの特徴量
        :param y_valid: バリデーションデータの目的変数
        """
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.array:
        """学習済のモデルでの予測値を返す

        :param X_test: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        pass

    @abstractmethod
    def save_model(self, path) -> None:
        """モデルの保存を行う"""
        pass

    @abstractmethod
    def load_model(self, path) -> None:
        """モデルの読み込みを行う"""
        pass

    @abstractmethod
    def plot_result(self):
        pass

class LGBMRegressor(Model):
    def __init__(self, num_boost_round: int=1000, early_stopping_rounds: int=20, verbose_eval=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.evals_result = {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval

    def train(self, X_train, y_train, X_valid, y_valid):
        mlflow.lightgbm.autolog()
        train_data = lgb.Dataset(X_train, label=y_train)
        validation_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        evals_data = [train_data, validation_data]
        evals_name = ['train', 'eval']
        
        self.model = lgb.train(
            params = self.params,
            train_set = train_data,
            valid_sets = evals_data,
            valid_names = evals_name,
            evals_result = self.evals_result,
            num_boost_round = self.num_boost_round,
            early_stopping_rounds = self.early_stopping_rounds,
            verbose_eval = self.verbose_eval,
            )

    def predict(self, X_test):
        test_data = X_test
        return self.model.predict(test_data, num_iteration=self.model.best_iteration)
    
    def plot_result(self):
        loss_train = self.evals_result['train']['rmse']
        loss_eval = self.evals_result['eval']['rmse']
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('RMSE')
        
        ax1.plot(loss_train, label='train loss')
        ax1.plot(loss_eval, label='test loss')
        
        plt.legend()
        plt.show()

class XGBRegressor(Model):
    def __init__(self, num_boost_round: int=1000, early_stopping_rounds: int=20, verbose_eval=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.evals_result = {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval

    def train(self, X_train, y_train, X_valid, y_valid):
        mlflow.xgboost.autolog()

        train_data = xgb.DMatrix(X_train, label=y_train)
        validation_data = xgb.DMatrix(X_valid, label=y_valid)
        evals = [(train_data,'train'),(validation_data,'eval')]

        self.model = xgb.train(
            self.params,
            train_data,
            evals = evals,
            num_boost_round = self.num_boost_round,
            early_stopping_rounds = self.early_stopping_rounds,
            verbose_eval = self.verbose_eval,
            evals_result = self.evals_result,
            )
        
    def predict(self, X_test):
        test_data = xgb.DMatrix(X_test)
        return self.model.predict(test_data)

    def plot_result(self):
        loss_train = self.evals_result['train']['rmse']
        loss_eval = self.evals_result['eval']['rmse']
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('RMSE')
        
        ax1.plot(loss_train, label='train loss')
        ax1.plot(loss_eval, label='test loss')
        
        plt.legend()
        plt.show()

class ModelNN(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット・スケーリング
        validation = va_x is not None
        scaler = StandardScaler()
        scaler.fit(tr_x)
        tr_x = scaler.transform(tr_x)
        tr_y = np_utils.to_categorical(tr_y, num_classes=9)

        if validation:
            va_x = scaler.transform(va_x)
            va_y = np_utils.to_categorical(va_y, num_classes=9)

        # パラメータ
        nb_classes = 9
        layers = self.params['layers']
        dropout = self.params['dropout']
        units = self.params['units']
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']

        # モデルの構築
        model = Sequential()
        model.add(Dense(units, input_shape=(tr_x.shape[1],)))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        for l in range(layers - 1):
            model.add(Dense(units))
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                           verbose=1, restore_best_weights=True)
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=128, verbose=2,
                      validation_data=(va_x, va_y), callbacks=[early_stopping])
        else:
            model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=128, verbose=2)

        # モデル・スケーラーの保持
        self.model = model
        self.scaler = scaler

    def predict(self, te_x):
        te_x = self.scaler.transform(te_x)
        pred = self.model.predict_proba(te_x)
        return pred

    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join('../model/model', f'{self.run_fold_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join('../model/model', f'{self.run_fold_name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Util.load(scaler_path)
