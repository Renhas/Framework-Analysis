import time
from typing import Any, Iterable, Tuple

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from src.loaders.loader import SaveLoader
from src.managers.fw_manager import TrialsConverter, ParamsManager, ManagersKit, Objective, ManagerConfig, FrameworkManager

class HyperoptParamsManager(ParamsManager):         
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        
    @staticmethod
    def extract_params(trial: dict):
        return {key: value[0] for key, value in trial['misc']["vals"].items()}
    
    def format_space(self, space: dict) -> dict:
        if isinstance(space["criterion"], int):
            space["criterion"] = self.params["criterion"][space["criterion"]]
        return {
            "criterion": space["criterion"],
            "max_depth": int(space["max_depth"]),
            "min_samples_leaf": int(space["min_samples_leaf"]),
            "min_samples_split": self.get_from_fraction(space)
        }
    
    def get_space(self) -> dict:
        current_space = {"criterion": hp.choice("criterion", self.params["criterion"])}
        for name in ["max_depth", "min_samples_leaf"]:
            current_space[name] = self.__get_uniformint(name)
        current_space["split_fraction"] = hp.uniform("split_fraction", 0, 1)
        return current_space
                
    def __get_uniformint(self, name):
        return scope.int(hp.quniform(name, self.params[name][0],
                                     self.params[name][1], self.params[name][2]))
    
class HyperoptConverter(TrialsConverter):
    def __init__(self, trials, params_manager: HyperoptParamsManager):
        self.trials = trials
        self.params_manager = params_manager
        
    def get_best_params(self) -> dict:
        return self.params_manager.format_space(self.trials.argmin)
    
    def _results_iterator(self) -> Iterable[dict]:
        for trial_result in self.trials.results:
            yield trial_result["cv_results"]
            
    def _trials_iterator(self) -> Iterable[Any]:
        return self.trials.trials
    
    def _one_to_dict(self, trial) -> dict:
        trial_duration = trial['refresh_time'] - trial['book_time']
        return {
            **self.params_manager.format_space(self.params_manager.extract_params(trial)),
            "Duration": trial_duration.total_seconds(),
            "Value": -1*trial["result"]["loss"]
        }
    
class HyperoptObjective(Objective):
    def __init__(self, managers_kit: ManagersKit) -> None:
        super().__init__(managers_kit)

    def objective(self, current_params: dict):
        cv_results, _ = self._inner_objective(current_params)
        return {"loss": -cv_results["test_r2"].mean(), "cv_results": cv_results,
                "status": STATUS_OK}
        
class HyperoptManager(FrameworkManager):
    def __init__(self, loader: SaveLoader, managers_kit: ManagersKit, config: ManagerConfig) -> None:
        super().__init__(loader, managers_kit, config)
        
    def _optimize(self) -> Tuple[TrialsConverter, int]:
        trials = Trials()
        start_time = time.time()
        fmin(HyperoptObjective(self.managers_kit).objective,
             space=self.managers_kit.params.get_space(),
             algo=tpe.suggest, trials=trials,
             max_evals=self.config.n_trials, verbose=False)
        iter_time = time.time() - start_time
        return HyperoptConverter(trials, self.managers_kit.params), iter_time