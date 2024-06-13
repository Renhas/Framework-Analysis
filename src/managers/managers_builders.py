from abc import ABC, abstractmethod
import copy
from typing import Iterable, Type

from sklearn.base import BaseEstimator

from src.loaders.loaders_factory import SaveLoaderFactory, OptunaSaveLoaderFactory
from src.managers.bayesian_manager import BayesianManager, BayesianParamsManager
from src.managers.talos_manager import TalosManager, TalosParamsManger
from src.managers.hyperopt_manager import HyperoptManager, HyperoptParamsManager
from src.managers.optuna_manager import OptunaManager, OptunaParamsManager
from src.managers.ray_tune_manager import RayTuneManager, RayTuneParamsManager
from src.managers.model_manager import InputData, ModelInfo, ModelManager
from src.managers.fw_manager import FrameworkManager, ManagerConfig, ManagersKit

class BuilderException(BaseException):
    pass    

class AModelManagerBuilder(ABC):
    def __init__(self) -> None:
        self._model_class = None
        self._train_data: InputData = None
        self._test_data: InputData = None
        
    @abstractmethod
    def set_model(self, model_class: Type[BaseEstimator]) -> "AModelManagerBuilder":
        pass
    
    @abstractmethod
    def set_train_data(self, x_data: Iterable, y_data: Iterable) -> "AModelManagerBuilder":
        pass 
    
    @abstractmethod
    def set_test_data(self, x_data: Iterable, y_data: Iterable) -> "AModelManagerBuilder":
        pass
    
    @abstractmethod
    def build(self) -> ModelManager:
        pass
    
    def _check(self) -> bool:
        return self._model_class is not None\
            and self._train_data.X is not None and self._train_data.Y is not None\
            and self._test_data.X is not None and self._test_data.Y is not None

class ModelManagerBuilder(AModelManagerBuilder):
    def set_model(self, model_class: BaseEstimator) -> AModelManagerBuilder:
        self._model_class = model_class
        return self
    
    def set_train_data(self, x_data: Iterable, y_data: Iterable) -> AModelManagerBuilder:
        self._train_data = InputData(x_data, y_data)
        return self
    
    def set_test_data(self, x_data: Iterable, y_data: Iterable) -> AModelManagerBuilder:
        self._test_data = InputData(x_data, y_data)
        return self
    
    def build(self) -> ModelManager:
        if self._test_data.X is None or self._test_data.Y is None:
            self._test_data = copy.deepcopy(self._train_data)
        if not self._check():
            raise BuilderException("Model class or train data not set!")
        return ModelManager(ModelInfo(self._model_class, self._train_data, self._test_data))
        
class AFrameworkManagerBuilder(ABC):
    def __init__(self, model_manager_builder: AModelManagerBuilder) -> None:
        self._mm_builder = model_manager_builder
        self._path: str = None
        self._params: dict = None
        self._config: ManagerConfig = None
        
    def set_model(self, model_class: Type[BaseEstimator]) -> "AFrameworkManagerBuilder":
        self._mm_builder.set_model(model_class)
        return self
    
    def set_train_data(self, x_data: Iterable, y_data: Iterable) -> "AFrameworkManagerBuilder":
        self._mm_builder.set_train_data(x_data, y_data)
        return self
    
    def set_test_data(self, x_data: Iterable, y_data: Iterable) -> "AFrameworkManagerBuilder":
        self._mm_builder.set_test_data(x_data, y_data)
        return self
    
    def set_path(self, path: str) -> "AFrameworkManagerBuilder":
        self._path = path
        return self
    
    def set_params(self, params: dict) -> "AFrameworkManagerBuilder":
        self._params = params
        return self
    
    def set_config(self, max_iter: int, n_trials: int) -> "AFrameworkManagerBuilder":
        self._config = ManagerConfig(max_iter, n_trials)
        return self
    
    def build(self) -> FrameworkManager:
        model_manager = self._mm_builder.build()
        if not self._check():
            raise BuilderException("Path, params or config doesn't set!")
        sl = self._get_sl_factory().create(self._path)
        kit = ManagersKit(model_manager, self._pm_class(self._params))
        return self._fm_class(sl, kit, self._config)  
    
    def _check(self):
        return self._path is not None and self._params is not None \
            and self._config.max_iter is not None\
            and self._config.n_trials is not None
    
    @abstractmethod
    def _get_sl_factory(self):
        pass
    
    @property
    @abstractmethod
    def _pm_class(self):
        pass
    
    @property
    @abstractmethod
    def _fm_class(self):
        pass
    
class RayTuneManagerBuilder(AFrameworkManagerBuilder):    
    def _get_sl_factory(self):
        return SaveLoaderFactory()
    
    @property
    def _pm_class(self):
        return RayTuneParamsManager
    
    @property
    def _fm_class(self):
        return RayTuneManager
    
class OptunaManagerBuilder(AFrameworkManagerBuilder):
    def _get_sl_factory(self):
        return OptunaSaveLoaderFactory()
    
    @property
    def _pm_class(self):
        return OptunaParamsManager
    
    @property
    def _fm_class(self):
        return OptunaManager

class HyperoptManagerBuilder(AFrameworkManagerBuilder):
    def _get_sl_factory(self):
        return SaveLoaderFactory()
    
    @property
    def _pm_class(self):
        return HyperoptParamsManager
    
    @property
    def _fm_class(self):
        return HyperoptManager
    
class BayesianManagerBuilder(AFrameworkManagerBuilder):
    def _get_sl_factory(self):
        return SaveLoaderFactory()
    
    @property
    def _pm_class(self):
        return BayesianParamsManager
    
    @property
    def _fm_class(self):
        return BayesianManager
    
class TalosManagerBuilder(AFrameworkManagerBuilder):
    def _get_sl_factory(self):
        return SaveLoaderFactory()
    
    @property
    def _pm_class(self):
        return TalosParamsManger
    
    @property
    def _fm_class(self):
        return TalosManager
