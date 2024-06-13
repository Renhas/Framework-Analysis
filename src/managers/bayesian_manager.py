from numbers import Number
import time
from typing import Any, Iterable, List, Tuple

from bayes_opt import BayesianOptimization, Events

from src.loaders.loader import SaveLoader
from src.managers.fw_manager import TrialsConverter, ParamsManager, ManagersKit, Objective, ManagerConfig, FrameworkManager

def my_round(x, base=5) -> int:
    return base * round(x/base)

class BayesianObserver():
    def __init__(self):
        self.history_time = []
        self.history_cv_res = []
        self.end_time = None
        
    @property
    def results(self) -> List[Tuple[int, dict]]:
        return zip(self.history_time, self.history_cv_res)
        
    def update(self, event: Events, instance: BayesianOptimization) -> None:
        self.history_time.append(time.time())
    
    def log_cv_res(self, cv_results: dict) -> None:
        self.history_cv_res.append(cv_results)
        
class BayesianParamsManager(ParamsManager):            
    def round_param(self, value: float, param_name: str):
        current_base = 1
        if len(self.params[param_name]) == 3 and isinstance(self.params[param_name][2], Number):
            current_base = self.params[param_name][2]
        return my_round(value, base=current_base)
        
    def get_space(self) -> dict:
        return {"criterion": (0, len(self.params["criterion"]) - 1),
                "max_depth": (self.params["max_depth"][0], self.params["max_depth"][1]),
                "min_samples_leaf": (self.params["min_samples_leaf"][0],
                                     self.params["min_samples_leaf"][1]),
                "split_fraction": (0, 1)}
    
    def format_space(self, current_params: dict) -> dict:
        rounded = {**{key: self.round_param(value, key) 
                   for key, value in current_params.items()
                   if key != "split_fraction"},
                   "split_fraction": current_params["split_fraction"]}
        rounded["criterion"] = self.params["criterion"][rounded["criterion"]]
        rounded["min_samples_split"] = self.get_from_fraction(rounded)
        rounded.pop("split_fraction")
        return rounded
    
class BayesianConverter(TrialsConverter):
    def __init__(self, optimizer: BayesianOptimization, observer: BayesianObserver,
                 params_manager: BayesianParamsManager):
        self.optimizer = optimizer
        self.observer = observer
        self.params_manager = params_manager
        
    def get_best_params(self) -> dict:
        return self.params_manager.format_space(self.optimizer.max["params"])
    
    def _one_to_dict(self, trial: Tuple[int, Any]) -> dict:
        idx, trial = trial
        duration = self.observer.history_time[idx + 1] - self.observer.history_time[idx]
        return {
            **self.params_manager.format_space(trial["params"]),
            "Duration": duration,
            "Value": trial["target"]
        }
        
    def _results_iterator(self) -> Iterable[dict]:
        return self.observer.history_cv_res
    
    def _trials_iterator(self) -> Iterable[Any]:
        return enumerate(self.optimizer.res[1:])
    
class BayesianObjective(Objective):
    def __init__(self, managers_kit: ManagersKit, observer: BayesianObserver) -> None:
        super().__init__(managers_kit)
        self.observer = observer

    def objective(self, **current_params: dict):
        cv_results, _ = self._inner_objective(current_params)
        self.observer.log_cv_res(cv_results)
        return cv_results["test_r2"].mean()
    
class BayesianManager(FrameworkManager):        
    def _optimize(self) -> Tuple[TrialsConverter, int]:
        observer = BayesianObserver()
        optimizer = BayesianOptimization(
            BayesianObjective(self.managers_kit, observer).objective,
            pbounds=self.managers_kit.params.get_space(),
            verbose=0,
            allow_duplicate_points=True)
        optimizer.subscribe(
            event=Events.OPTIMIZATION_STEP,
            subscriber=observer,
            callback=None
        )
        start_time = time.time()
        optimizer.maximize(init_points=1, n_iter=self.config.n_trials)
        iter_time = time.time() - start_time
        return BayesianConverter(optimizer, observer, self.managers_kit.params), iter_time