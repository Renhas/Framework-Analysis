from copy import deepcopy
import time
from typing import Iterable

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator

class InputData():
    def __init__(self, x_data, y_data):
        self.__x_data = deepcopy(x_data)
        self.__y_data = deepcopy(y_data)
        
    @property
    def X(self):
        return deepcopy(self.__x_data)
    
    @property
    def Y(self):
        return deepcopy(self.__y_data)

class ModelInfo():
    def __init__(self, model_class, train_data: InputData, test_data: InputData):
        self.__model_class = model_class
        self.__train_data = train_data
        self.__test_data = test_data
        
    @property
    def model_class(self):
        return self.__model_class

    @property
    def train_data(self):
        return self.__train_data
    
    @property
    def test_data(self):
        return self.__test_data

class ModelManager():
    def __init__(self, info: ModelInfo) -> None:
        self.__info = info
        
    @property
    def info(self) -> ModelInfo:
        return self.__info
        
    def create_model(self, params: dict) -> BaseEstimator:
        return self.info.model_class(**params)
    
    def train_model(self, model: BaseEstimator):
        return model.fit(self.info.train_data.X, self.info.train_data.Y)
        
    def test_model(self, trained_model: BaseEstimator):
        start = time.time()
        predicted = trained_model.predict(self.info.test_data.X)
        end = time.time() - start
        return {("Time", "Test"): end,
                ("MAE", "Test"): mean_absolute_error(self.info.test_data.Y, predicted),
                ("R2", "Test"): r2_score(self.info.test_data.Y, predicted)}
        
    def cross_validate_model(self, model: BaseEstimator, metrics: Iterable = ("neg_mean_absolute_error", "r2"), n_jobs=-1) -> dict:
        return cross_validate(model, self.info.train_data.X, self.info.train_data.Y,
                              n_jobs=n_jobs, return_train_score=True,
                              scoring=metrics)
    
class CVConverter():
    cv_results_keys = ("Time", "Train"), ("Time", "Val"), \
                      ("MAE", "Train"), ("MAE", "Val"), \
                      ("R2", "Train"), ("R2", "Val")
    
    @staticmethod    
    def format_cv_results(cv_results: dict) -> dict:
        return {
            ("Time", "Train"): cv_results["fit_time"].mean(),
            ("Time", "Val"): cv_results["score_time"].mean(),
            ("MAE", "Train"): cv_results["train_neg_mean_absolute_error"].mean() * -1,
            ("MAE", "Val"): cv_results["test_neg_mean_absolute_error"].mean() * -1,
            ("R2", "Train"): cv_results["train_r2"].mean(),
            ("R2", "Val"): cv_results["test_r2"].mean(),
        }

    