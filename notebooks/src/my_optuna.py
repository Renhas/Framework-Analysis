import joblib
from src.loader import SaveLoader, PathManager, Saver, Loader

class PathManagerOptuna(PathManager):
    def __init__(self, main_folder, results_path):
        super().__init__(main_folder, results_path)
        self.studies = f"{self.folder}studies/"
        self.make_dirs("studies")
        
    def one_study(self, idx: int):
        return f"{self.studies}study_{idx}.joblib"

class OptunaSaver(Saver):
    def __init__(self, path_manager: PathManager, iter_count: int) -> None:
        super().__init__(path_manager, iter_count)
    
    def save_study(self, study):
        joblib.dump(study, self._paths.one_study(self.iter_count),
                    compress=3)

class OptunaLoader(Loader):
    def __init__(self, path_manager: PathManager, iter_count: int) -> None:
        super().__init__(path_manager, iter_count)
    
    def load_study(self, idx: int):
        self._idx_check(idx)
        return joblib.load(self._paths.one_study(idx))

class OptunaSaveLoader(SaveLoader):
    def __init__(self, path_manager: PathManager):
        super().__init__(path_manager)
        self.saver = self._create_saver(path_manager)
        self.loader = self._create_loader(path_manager)
        
    def save_study(self):
        self.saver.save_study()
        
    def load_study(self, idx: int):
        return self.loader.load_study(idx)
        
    def _create_saver(self, path_manager):
        return OptunaSaver(path_manager, self.iter_count)

    def _create_loader(self, path_manager):
        return OptunaLoader(path_manager, self.iter_count)