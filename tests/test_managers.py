
import os
import logging
import shutil

import ray
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from src.managers.hyperopt_manager import HyperoptManager, HyperoptParamsManager
from src.managers.optuna_manager import OptunaManager, OptunaParamsManager
from src.loaders.loaders_factory import SaveLoaderFactory, OptunaSaveLoaderFactory
from src.managers.ray_tune_manager import RayTuneManager, RayTuneParamsManager
from src.managers.model_manager import InputData, ModelInfo, ModelManager
from src.managers.fw_manager import ManagerConfig, ManagersKit

def create_model_manager():
    x_data, y_data = load_diabetes(return_X_y=True, as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, 
                                                        random_state = 2024)
    return ModelManager(ModelInfo(DecisionTreeRegressor, InputData(x_train, y_train), InputData(x_test, y_test)))

def get_params():
    return {
        "criterion": ['squared_error', 'friedman_mse', 'poisson'],
        "max_depth": [5, 60, 5], 
        "min_samples_split": [2, 20, 1],
        "min_samples_leaf": [1, 10, 1]
    }
    
def get_config():
    return ManagerConfig(max_iter=3, n_trials=2)

def clean_up(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        
class TestRayTune():
    def test_search(self):
        clean_up("tests/TestRayTune/")
        sl = SaveLoaderFactory().create("tests/TestRayTune/")
        kit = ManagersKit(create_model_manager(), RayTuneParamsManager(get_params()))
        config = get_config()
        manager = RayTuneManager(sl, kit, config)
        
        os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
        if ray.is_initialized():
            ray.shutdown()
        ray.init(ignore_reinit_error=True, _temp_dir=os.path.abspath("./Temp/Ray"), logging_level=logging.ERROR)
        manager.search()
                
        assert sl.iter_count == config.max_iter
        assert sl.load_trials(0).shape[0] == config.n_trials
        clean_up("tests/TestRayTune/")
        
class TestOptuna():
    def test_search(self):
        clean_up("tests/TestOptuna/")
        sl = OptunaSaveLoaderFactory().create("tests/TestOptuna/")
        kit = ManagersKit(create_model_manager(), OptunaParamsManager(get_params()))
        config = get_config()
        manager = OptunaManager(sl, kit, config)
        
        manager.search()
        
        assert sl.iter_count == config.max_iter
        assert sl.load_trials(0).shape[0] == config.n_trials
        clean_up("tests/TestOptuna/")
        
class TestHyperopt():
    def test_search(self):
        clean_up("tests/TestHyperopt/")
        sl = SaveLoaderFactory().create("tests/TestHyperopt/")
        kit = ManagersKit(create_model_manager(), HyperoptParamsManager(get_params()))
        config = get_config()
        manager = HyperoptManager(sl, kit, config)
        
        manager.search()
        
        assert sl.iter_count == config.max_iter
        assert sl.load_trials(0).shape[0] == config.n_trials
        clean_up("tests/TestHyperopt/")
        