import os
import logging
import shutil
from typing import Callable, Type

import pytest
import ray
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from src.managers.managers_builders import (
    AFrameworkManagerBuilder, BuilderException, ModelManagerBuilder, RayTuneManagerBuilder, OptunaManagerBuilder,
    HyperoptManagerBuilder, BayesianManagerBuilder, TalosManagerBuilder
)

def get_params():
    return {
        "criterion": ['squared_error', 'friedman_mse', 'poisson'],
        "max_depth": [5, 60, 5], 
        "min_samples_split": [2, 20, 1],
        "min_samples_leaf": [1, 10, 1]
    }

class DataForTests():
    def __init__(self) -> None:
        x_data, y_data = load_diabetes(return_X_y=True, as_frame=True)
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x_data, y_data, test_size = 0.2, random_state = 2024)
    
    @property
    def train(self):
        return self.x_train, self.y_train
    
    @property
    def test(self):
        return self.x_test, self.y_test 

class TestModelManagerBuilders():
    @pytest.mark.parametrize(
        ("model", "x_train", "y_train", "x_test", "y_test", "expected_test"), [
            (DecisionTreeRegressor, DataForTests().train[0], DataForTests().train[1],
             DataForTests().test[0], DataForTests().test[1], DataForTests().test),
            (DecisionTreeRegressor, DataForTests().train[0], DataForTests().train[1],
             None, None, DataForTests().train)
        ]
    )
    def test_create(self, model, x_train, y_train, x_test, y_test, expected_test):
        builder = ModelManagerBuilder()
        builder.set_model(model).set_train_data(x_train, y_train)\
            .set_test_data(x_test, y_test)
        manager = builder.build()
        
        assert isinstance(manager.create_model({}), DecisionTreeRegressor)
        assert manager.info.train_data.X.equals(x_train)
        assert manager.info.train_data.Y.equals(y_train)
        assert manager.info.test_data.X.equals(expected_test[0])
        assert manager.info.test_data.Y.equals(expected_test[1])
        
    @pytest.mark.xfail(raises=BuilderException, strict=True)
    @pytest.mark.parametrize(
        ("model", "x_train", "y_train", "x_test", "y_test"), [
            (DecisionTreeRegressor, None, None, None, None),
            (None, None, None, None, None),
            (None, None, None, DataForTests().test[0], DataForTests().test[1])
        ]
    )
    def test_fall(self, model, x_train, y_train, x_test, y_test):
        builder = ModelManagerBuilder()
        builder.set_model(model).set_train_data(x_train, y_train)\
            .set_test_data(x_test, y_test)
        builder.build()

def set_data(builder: AFrameworkManagerBuilder):
    builder.set_model(DecisionTreeRegressor).set_train_data(*DataForTests().train)\
           .set_test_data(*DataForTests().test)

def clean_up(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        
class TestFrameworkManagersBuilders():
    @pytest.mark.parametrize(
        "builder_type", [RayTuneManagerBuilder, OptunaManagerBuilder,
                         HyperoptManagerBuilder, BayesianManagerBuilder,
                         TalosManagerBuilder]
    )
    @pytest.mark.parametrize("max_iter", [1, 3, 5])
    @pytest.mark.parametrize("n_trials", [2, 4, 6])
    def test_create(self, builder_type: Type[AFrameworkManagerBuilder],
                    max_iter: int, n_trials: int):
        builder = builder_type(ModelManagerBuilder())
        set_data(builder)
        builder.set_config(max_iter, n_trials).set_params(get_params()).set_path("tests/TestBuilders/")
        manager = builder.build()
        
        assert manager.config.max_iter == max_iter
        assert manager.config.n_trials == n_trials
        assert manager.loader.iter_count == 0
        assert manager.managers_kit.params.params == get_params()
        clean_up("tests/TestBuilders/")
        
    @pytest.mark.xfail(raises=BuilderException, strict=True)
    @pytest.mark.parametrize(
        "builder_type", [RayTuneManagerBuilder, OptunaManagerBuilder,
                         HyperoptManagerBuilder, BayesianManagerBuilder,
                         TalosManagerBuilder]
    )
    @pytest.mark.parametrize(
        ("max_iter", "n_trials", "params"), [
            (1, None, get_params()), 
            (None, 2, get_params()),
            (1, 2, None)
        ]
    )
    def test_fall(self, builder_type: Type[AFrameworkManagerBuilder],
                  max_iter: int, n_trials: int, params: dict):
        builder = builder_type(ModelManagerBuilder())
        set_data(builder)
        builder.set_config(max_iter, n_trials).set_params(params).set_path("tests/TestBuilders/")
        builder.build()

def get_config_data():
    return 3, 2

def set_managers_data(builder: AFrameworkManagerBuilder):
    set_data(builder)
    builder.set_config(*get_config_data()).set_params(get_params())\
           .set_path("tests/TestManagers/")

def ray_tune_prepare():
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, _temp_dir=os.path.abspath("./Temp/Ray"), logging_level=logging.ERROR) 
    
def ray_tune_clean():
    if ray.is_initialized():
        ray.shutdown()
    clean_up("Temp/")
    
class TestManagers():
    @pytest.mark.parametrize(
        ("builder_type", "prepare_func", "clean_func"), [
            (RayTuneManagerBuilder, ray_tune_prepare, ray_tune_clean),
            (OptunaManagerBuilder, None, None),
            (HyperoptManagerBuilder, None, None),
            (BayesianManagerBuilder, None, None),
            (TalosManagerBuilder, None, lambda: clean_up("TalosSeek")),
            
        ]
    )
    def test_search(self, builder_type: Type[AFrameworkManagerBuilder],
                    prepare_func: Callable, clean_func: Callable):
        clean_up("tests/TestManagers/")
        builder = builder_type(ModelManagerBuilder())
        set_managers_data(builder)
        manager = builder.build()
        
        if prepare_func:
            prepare_func()
        manager.search()
                
        assert manager.loader.iter_count == get_config_data()[0]
        assert manager.loader.load_trials(0).shape[0] == get_config_data()[1]
        if clean_func:
            clean_func()
        clean_up("tests/TestManagers/")
