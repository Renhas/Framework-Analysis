import os
import pandas as pd
import joblib

class PathManager():
    def __init__(self, folder_path):
        self.folder = folder_path
        self.models_folder = f"{self.folder}models/"
        self.trials_folder = f"{self.folder}trials/"
        self.results = f"{self.folder}results.csv"
        self.make_dirs("models", "trials")
        
    @property
    def trials_count(self):
        return len([name for name in os.listdir(self.trials_folder) 
                    if os.path.isfile(os.path.join(self.trials_folder, name))])
        
    def one_trials(self, idx: int):
        return f"{self.trials_folder}trials_{idx}.csv"
    
    def one_model(self, idx: int):
        return f"{self.models_folder}model_{idx}.joblib"
    
    def make_dirs(self, *folders):
        for folder in folders:
            if not os.path.exists(f"{self.folder}{folder}"):
                os.makedirs(f"{self.folder}{folder}")
    
class Loader():
    def __init__(self, path_manager: PathManager) -> None:
        self._paths = path_manager
        self.iter_count = path_manager.trials_count
        
    def load_results(self):
        if self.iter_count <= 0:
            raise FileNotFoundError("results.csv not exists")
        return pd.read_csv(self._paths.results, header=[0, 1])

    def load_trials(self, idx: int):
        self._idx_check(idx)
        return pd.read_csv(self._paths.one_trials(idx))

    def load_model(self, idx: int):
        self._idx_check(idx)
        return joblib.load(self._paths.one_model(idx))
    
    def _idx_check(self, idx: int):
        if idx < 0 or idx >= self.iter_count:
            raise IndexError(f"idx out of range [0, {self.iter_count - 1}]")
                
class Saver():
    def __init__(self, loader: Loader) -> None:
        self._paths = loader._paths
        self.iter_count = loader.iter_count
        self.loader = loader
        
    def save_res(self, data: dict):
        self.__form_results(data).to_csv(self._paths.results, index=False)

    def save_trial(self, trial_history: dict):
        pd.DataFrame(trial_history)\
          .to_csv(self._paths.one_trials(self.iter_count), index=False)

    def save_model(self, model):
        joblib.dump(model, self._paths.one_model(self.iter_count),
                    compress=3)
        
    def __form_results(self, new_results: dict) -> pd.DataFrame:
        new_dt = pd.DataFrame({key: [value] for key, value in new_results.items()})
        if self.iter_count > 0:
            current_dt = self.loader.load_results()
            new_dt = pd.concat([current_dt, new_dt], axis=0)\
                       .reset_index(drop=True)
        return new_dt
                
class SaveLoader():
    def __init__(self, loader: Loader, saver: Saver):
        self.__iter_count = loader.iter_count
        self.saver = saver
        self.loader = loader

    @property
    def iter_count(self):
        return self.__iter_count
    
    def next_iter(self):
        self.__iter_count += 1
        self.saver.iter_count = self.iter_count
        self.loader.iter_count = self.iter_count
    
    def save_res(self, data: dict):
        self.saver.save_res(data)

    def save_trial(self, trial_history: dict):
        self.saver.save_trial(trial_history)

    def save_model(self, model):
        self.saver.save_model(model)

    def load_results(self):
        return self.loader.load_results()

    def load_trials(self, idx: int):
        return self.loader.load_trials(idx)

    def load_model(self, idx: int):
        return self.loader.load_model(idx)