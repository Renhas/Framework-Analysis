import copy
import time
from typing import Any, Iterable, Tuple

import numpy as np
import talos

from src.loaders.loader import SaveLoader
from src.managers.fw_manager import TrialsConverter, ParamsManager, ManagersKit, Objective, ManagerConfig, FrameworkManager

class TalosHistory():
    def __init__(self, cv_results) -> None:
        self.history = {"val_cv_results": [cv_results],
                        "metric": [cv_results["test_r2"].mean()],
                        "val_temp": [np.NaN]}

class TalosParamsManger(ParamsManager):        
    def get_space(self) -> dict:
        return {"criterion": self.params["criterion"],
                "max_depth": self.__get_param_range("max_depth"),
                "min_samples_leaf": self.__get_param_range("min_samples_leaf"),
                "split_fraction": (0., 1.1, 11)}      
    
    def format_space(self, space: dict) -> dict:
        return {
            "criterion": space["criterion"],
            "max_depth": space["max_depth"],
            "min_samples_leaf": space["min_samples_leaf"],
            "min_samples_split": self.get_from_fraction(space)
        }
          
    def __get_param_range(self, param_name: str) -> Tuple[int, int, int]:
        start = self.params[param_name][0]
        end = self.params[param_name][1] + self.params[param_name][2]
        step = (end - start) // self.params[param_name][2]
        return (start, end, step)
    
class TalosConverter(TrialsConverter):
    def __init__(self, scan_obj: talos.Scan, params_manager: TalosParamsManger):
        self.scan_obj = scan_obj
        self.params_manager = params_manager
        
    def get_best_params(self) -> dict:
        best_trial = self.scan_obj.data.loc[self.scan_obj.data["metric"].idxmax()]
        return self.params_manager.format_space(self.__trial_params(best_trial))
    
    def _one_to_dict(self, trial) -> dict:
        return {
            **self.params_manager.format_space(self.__trial_params(trial)),
            "Duration": trial["duration"],
            "Value": trial["metric"]
        }
        
    def _results_iterator(self) -> Iterable[dict]:
        for _, row in self.scan_obj.data.iterrows():
            yield row["val_cv_results"]
            
    def _trials_iterator(self) -> Iterable[Any]:
        for _, row in self.scan_obj.data.iterrows():
            yield row
        
    def __trial_params(self, trial) -> dict:
        return {key: trial[key] for key in self.scan_obj.params.keys()}       

class TalosObjective(Objective):
    def objective(self, x_train, y_train, x_val, y_val, current_params):
        cv_results, model = self._inner_objective(current_params)
        return TalosHistory(cv_results), model

class TalosManager(FrameworkManager):        
    def _optimize(self) -> Tuple[TrialsConverter, int]:
        train_data = self.managers_kit.model.info.train_data
        start_time = time.time()
        scan_obj = talos.Scan(copy.deepcopy(train_data.X.values), 
                              copy.deepcopy(train_data.Y.values),
                              self.managers_kit.params.get_space(),
                              TalosObjective(self.managers_kit).objective,
                              "TalosSeek", val_split=0, round_limit=self.config.n_trials, 
                              reduction_metric="metric",
                              save_weights=False, clear_session=False,
                              disable_progress_bar=True)
        iter_time = time.time() - start_time
        return TalosConverter(scan_obj, self.managers_kit.params), iter_time