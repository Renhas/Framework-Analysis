import joblib
from src.loaders.loader import SaveLoader, PathManager, Saver, Loader

class OptunaPathManager(PathManager):
    def __init__(self, folder_path: str):
        super().__init__(folder_path)
        self.studies = f"{self.folder}studies/"
        self.make_dirs("studies")
        
    def one_study(self, idx: int):
        return f"{self.studies}study_{idx}.joblib"

class OptunaLoader(Loader):
    def __init__(self, path_manager: OptunaPathManager) -> None:
        super().__init__(path_manager)
    
    def load_study(self, idx: int):
        self._idx_check(idx)
        return joblib.load(self._paths.one_study(idx))

class OptunaSaver(Saver):
    def __init__(self, loader: OptunaLoader) -> None:
        super().__init__(loader)
    
    def save_study(self, study):
        joblib.dump(study, self._paths.one_study(self.iter_count),
                    compress=3)

class OptunaSaveLoader(SaveLoader):
    def __init__(self, folder_path: str):
        path_manager = OptunaPathManager(folder_path)
        loader = OptunaLoader(path_manager)
        super().__init__(loader, OptunaSaver(loader))
        
    def save_study(self, study):
        self.saver.save_study(study)
        
    def load_study(self, idx: int):
        return self.loader.load_study(idx)
