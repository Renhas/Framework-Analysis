from abc import ABC, abstractmethod

from src.loaders.optuna_loader import (
    PathManager, Loader, Saver, SaveLoader, OptunaSaveLoader
)

class ASaveLoaderFactory(ABC):
    @abstractmethod
    def create(self, folder_path) -> SaveLoader:
        pass

class SaveLoaderFactory(ASaveLoaderFactory):
    def create(self, folder_path) -> SaveLoader:
        loader = Loader(PathManager(folder_path))
        return SaveLoader(loader, Saver(loader))
    
class OptunaSaveLoaderFactory(ASaveLoaderFactory):
    def create(self, folder_path) -> OptunaSaveLoader:
        return OptunaSaveLoader(folder_path)