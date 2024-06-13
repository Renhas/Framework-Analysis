import copy
from typing import Any, Iterable, Tuple
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator

from src.loaders.loader import SaveLoader
from src.managers.model_manager import ModelManager, CVConverter

class Results():
    def __init__(self, dict_keys: Iterable):
        self.__values = {key: [] for key in dict_keys}
    
    @property
    def values(self) -> dict:
        return copy.deepcopy(self.__values)
    
    @property
    def mean(self) -> dict:
        return {key: sum(value)/len(value) for key, value in self.__values.items()}

    def add(self, new_dict: dict) -> None:
        for key, value in new_dict.items():
            self.__values[key].append(value)

class Logger(ABC):
    def __init__(self):
        pass
    
    def log(self, message: str):
        pass

class TrialsConverter(ABC):
    def to_results(self):
        results = Results(CVConverter.cv_results_keys)
        for result in self._results_iterator():
            cv_results = CVConverter.format_cv_results(result)
            results.add(cv_results)
        best_params = self.get_best_params()
        result = results.mean
        result.update({("Params", key): value for key, value in best_params.items()})
        return result

    def to_history(self, params_names: list):
        results = Results(["Duration", "Value", *params_names])
        for trial in self._trials_iterator():
            trial_dict = self._one_to_dict(trial)
            results.add(trial_dict)
        return results.values
    
    def format_all(self, params_names: list, iter_time: int):
        trials_results = self.to_results()
        trials_results[("Time", "Iteration")] = iter_time
        return self.to_history(params_names), trials_results, self.get_best_params()

    @abstractmethod
    def get_best_params(self) -> dict:
        pass
    
    @abstractmethod
    def _results_iterator(self):
        pass
    
    @abstractmethod
    def _trials_iterator(self):
        pass
        
    @abstractmethod
    def _one_to_dict(self, trial) -> dict:
        pass

class ParamsManager(ABC):
    def __init__(self, params: dict) -> None:
        self.params = params
        
    def get_from_fraction(self, current_params: dict):
        leaf_value = current_params['min_samples_leaf'] * 2
        leaf_value *= 1 - current_params["split_fraction"]
        split_value = self.params["min_samples_split"][1]
        split_value *= current_params["split_fraction"]
        return int(leaf_value + split_value)
    
    @abstractmethod
    def get_space(self) -> dict:
        pass
    
    @abstractmethod
    def format_space(self, space: dict) -> dict:
        pass

class ManagersKit(ABC):
    def __init__(self, model_manager: ModelManager, params_manager: ParamsManager) -> None:
        self.model = model_manager
        self.params = params_manager

class Objective(ABC):
    def __init__(self, managers_kit: ManagersKit) -> None:
        self.kit = managers_kit
        
    def _inner_objective(self, params: dict, n_jobs=-1):
        current_params = self.kit.params.format_space(params)
        model = self.kit.model.create_model(current_params)
        cv_results = self.kit.model.cross_validate_model(model, n_jobs=n_jobs)
        return cv_results, model
    
    @abstractmethod
    def objective(self, *args, **kwargs):
        pass

class ManagerConfig():
    def __init__(self, max_iter: int, n_trials: int) -> None:
        self.max_iter = max_iter
        self.n_trials = n_trials

class FrameworkManager(ABC):
    def __init__(self, loader: SaveLoader, managers_kit: ManagersKit, config: ManagerConfig) -> None:
        self.__kit = managers_kit
        self.__loader = loader
        self.config = config
    
    @property
    def loader(self) -> SaveLoader:
        return self.__loader
    
    @property
    def managers_kit(self) -> ManagersKit:
        return self.__kit
        
    def search(self, logger: Logger = None) -> None:
        while self.__loader.iter_count < self.config.max_iter:
            if logger:
                logger.log(f"Iter #{self.__loader.iter_count + 1}: ")
            self.__save_all(*self.__get_results())
            self.__loader.next_iter()
            if logger:
                logger.log("Results saved")
                
    def __get_results(self) -> Tuple[BaseEstimator, dict, dict]:
        history, results, best_params = self.one_iteration()
        model, test_results = self.__best_model(best_params)
        results.update(test_results)
        return model, history, results

    def one_iteration(self) -> Tuple[dict, dict, dict]:
        return self.__format_results(*self._optimize())

    @abstractmethod            
    def _optimize(self) -> Tuple[TrialsConverter, int]: 
        pass
    
    def __save_all(self, model: BaseEstimator, history: dict, results: dict):
        self.__loader.save_model(model)
        self.__loader.save_trial(history)
        self.__loader.save_res(results)
        
    def __best_model(self, best_params: dict) -> Tuple[BaseEstimator, dict]:
        model = self.__kit.model.create_model(best_params)
        trained = self.__kit.model.train_model(model)
        return trained, self.__kit.model.test_model(trained)
    
    def __format_results(self, converter: TrialsConverter, iter_time: int):
        return converter.format_all(self.__kit.params.params.keys(), iter_time)
    