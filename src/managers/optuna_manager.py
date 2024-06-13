import os
import time

import optuna

from src.loaders.optuna_loader import OptunaSaveLoader
from src.loaders.loader import SaveLoader
from src.managers.model_manager import ModelManager
from src.managers.fw_manager import FrameworkManager, ManagerConfig, ManagersKit, TrialsConverter, ParamsManager, Objective

class OptunaConverter(TrialsConverter):
    def __init__(self, study):
        self.study = study
    
    def get_best_params(self) -> dict:
        return self.study.best_params
    
    def _results_iterator(self):
        for trial in self.study.get_trials():
            yield trial.user_attrs["cv_results"]
            
    def _trials_iterator(self):
        return self.study.get_trials()
        
    def _one_to_dict(self, trial) -> dict:
        return {
            **trial.params,
            "Duration": trial.duration.total_seconds(),
            "Value": trial.values[0]
        }

class OptunaParamsManager(ParamsManager):    
    def get_space(self) -> dict:
        pass
    
    def format_space(self, space: dict) -> dict:
        return space
    
class OptunaObjective(Objective):        
    def __suggest_int(self, trial, name: str):
        return trial.suggest_int(name,
                                 low=self.kit.params.params[name][0],
                                 high=self.kit.params.params[name][1],
                                 step=self.kit.params.params[name][2])
        
    def __suggest_except_last(self, trial):
        current_params = {
            "criterion": trial.suggest_categorical("criterion", 
                                                   self.kit.params.params["criterion"])
        }
        for name in ["max_depth", "min_samples_leaf"]:
            suggested = self.__suggest_int(trial, name)
            current_params[name] = suggested
        return current_params
    
    def __suggest_all(self, trial):
        current_params = self.__suggest_except_last(trial)
        current_params["min_samples_split"] = trial.suggest_int(
            "min_samples_split",
            low=current_params["min_samples_leaf"] * 2,
            high=self.kit.params.params["min_samples_split"][1],
            step=self.kit.params.params["min_samples_split"][2]
        )
        return current_params

    def objective(self, trial):
        current_params = self.__suggest_all(trial)
        cv_results, _ = self._inner_objective(current_params)
        trial.set_user_attr("cv_results", cv_results)
        return cv_results["test_r2"].mean()
    
class OptunaManager(FrameworkManager):        
    def _optimize(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        start_time = time.time()
        study.optimize(OptunaObjective(self.managers_kit).objective,
                       n_trials=self.config.n_trials, n_jobs=-1)
        iter_time = time.time() - start_time
        self.loader.save_study(study)
        return OptunaConverter(study), iter_time
    