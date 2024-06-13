import gc
import os
import random
import time

import ray
from ray import train, tune

from src.loaders.loader import SaveLoader
from src.managers.model_manager import ModelManager
from src.managers.fw_manager import FrameworkManager, ManagerConfig, ManagersKit, TrialsConverter, ParamsManager, Objective

class RayTuneConverter(TrialsConverter):
    def __init__(self, results_grid):
        self.results_grid = results_grid
        
    def get_best_params(self) -> dict:
        return self.results_grid.get_best_result().metrics["config"]
    
    def _results_iterator(self):
        for trial_result in self.results_grid._results:
            yield trial_result.metrics["cv_results"]
    
    def _trials_iterator(self):
        for trial_result in self.results_grid._results:
            yield trial_result.metrics
    
    def _one_to_dict(self, trial) -> dict:
        return {
            **trial["config"],
            "Duration": trial["time_total_s"],
            "Value": trial["score"]
        }
        
class RayTuneParamsManager(ParamsManager):
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        
    def __get_randint(self, name: str):
        return tune.qrandint(self.params[name][0], self.params[name][1], self.params[name][2])
    
    def __sample_split(self, config: dict):
        return random.randrange(config["min_samples_leaf"] * 2,
                                self.params["min_samples_split"][1],
                                self.params["min_samples_split"][2])
    
    def get_space(self) -> dict:
        current_space = {"criterion": tune.choice(self.params["criterion"])}
        for name in ["max_depth", "min_samples_leaf"]:
            current_space[name] = self.__get_randint(name)
        current_space["min_samples_split"] = tune.sample_from(self.__sample_split)
        return current_space
    
    def format_space(self, space: dict) -> dict:
        return space

class RayTuneObjective(Objective):
    def __init__(self, managers_kit: ManagersKit) -> None:
        super().__init__(managers_kit)

    def objective(self, config: dict):
        cv_results, model = self._inner_objective(config, n_jobs=1)
        del model
        gc.collect()
        return {"score": cv_results["test_r2"].mean(), "cv_results": cv_results}
    
class CustomReporter(tune.ProgressReporter):
    def should_report(self, trials, done=False):
        return False
    def report(self, trials, done, *sys_info):
        pass
    
class RayTuneManager(FrameworkManager):
    def __init__(self, loader: SaveLoader, managers_kit: ManagersKit, config: ManagerConfig) -> None:
        super().__init__(loader, managers_kit, config)
    
    def _optimize(self):
        tuner = tune.Tuner(RayTuneObjective(self.managers_kit).objective,
                           param_space=self.managers_kit.params.get_space(),
                           tune_config=self.__tune_config(),
                           run_config=self.__run_config())
        start_time = time.time()
        result_grid = tuner.fit()
        iter_time = time.time() - start_time
        return RayTuneConverter(result_grid), iter_time
    
    def __tune_config(self):
        return tune.TuneConfig(metric="score", mode="max",
                               num_samples=self.config.n_trials,
                               max_concurrent_trials=None)
    
    def __run_config(self):
        return train.RunConfig(verbose=0, progress_reporter=CustomReporter(),
                               storage_path=os.path.abspath("./Temp/Ray/Storage"),
                               checkpoint_config=self.__checkpoint_config(),
                               failure_config=self.__failure_config())
    
    @staticmethod
    def __checkpoint_config():
        return train.CheckpointConfig(num_to_keep=1, checkpoint_frequency=0)
    
    @staticmethod
    def __failure_config():
        return train.FailureConfig(max_failures=3)
    