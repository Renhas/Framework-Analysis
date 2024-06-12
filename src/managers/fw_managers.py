import copy
from abc import ABC, abstractmethod
from .loader import SaveLoader

class Results():
    def __init__(self, dict_keys):
        self.__values = {key: [] for key in dict_keys}
    
    @property
    def values(self) -> dict:
        return copy.deepcopy(self.__values)

    def add(self, new_dict: dict):
        for key, value in new_dict.items():
            self.__values[key].append(value)
    
    @property
    def mean(self):
        return {key: sum(value)/len(value) for key, value in self.__values.items()}
    
class FrameworkManager(ABC):
    def __init__(self, model_class, loader: SaveLoader):
        self.__model_class = model_class
        self.__loader = loader
    
    @property
    def loader(self):
        return self.__loader

    def create_model(self, *args, **kwargs):
        return self.__model_class(*args, **kwargs)

    @abstractmethod
    def one_iteration(self):
        pass
    
    def __save_all(self, model, trial_dict, data_dict):
        self.__loader.save_model(model)
        self.__loader.save_trial(trial_dict)
        self.__loader.save_res(data_dict)
        
    def __best_model(self, best_params):
        model = self.create_model(**best_params)
        model.fit(x_train, y_train)
        return model, test_model(model)
        
    def search(self, max_iter):
        while self.__loader.iter_count < max_iter:
            print(f"Iter #{self.__loader.iter_count + 1}: ", end="")
            history, results, best_params = self.one_iteration()
            model, test_results = self.__best_model(best_params)
            results.update(test_results)
            self.__save_all(model, history, results)
            self.__loader.next_iter()
            print("Results saved")