import os
import random
import shutil
import time

import pytest
import pandas as pd

from src.loaders.loaders_factory import SaveLoaderFactory, OptunaSaveLoaderFactory
from src.loaders.optuna_loader import OptunaPathManager
from src.loaders.loader import PathManager, SaveLoader

class EmptyTestModel:
    def __init__(self, number) -> None:
        self.number = number
        

def clean_up(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)

class TestBaseLoader:
    @pytest.mark.parametrize(
        ("path", "expected"), [
            ("tests/Test/", "tests/Test/"),
            ("tests/Test", "tests/Test/")
        ]
    )
    def test_path_manager(self, path, expected):
        clean_up(expected)
        manager = PathManager(path)
        assert os.path.exists(expected)
        assert os.path.exists(f"{expected}models/")
        assert os.path.exists(f"{expected}trials/")
        assert manager.trials_count == 0
        assert manager.one_trials(0) == f"{expected}trials/trials_0.csv"
        assert manager.one_model(0) == f"{expected}models/model_0.joblib"
        clean_up(expected)
    
    def test_saveloader(self):
        clean_up("tests/Test/")
        sl = SaveLoaderFactory().create("tests/Test/")
        iter_count = 5
        expected = self.__get_expected(sl, iter_count)
        
        assert sl.iter_count == iter_count
        for iter_id in range(iter_count):
            assert sl.load_model(iter_id).number == expected["models"][iter_id].number    
            assert sl.load_trials(iter_id).equals(pd.DataFrame(expected["trials"][iter_id]))
        assert sl.load_results().equals(pd.DataFrame(expected["res"]))
        clean_up("tests/Test/")
        
    def __get_expected(self, sl: SaveLoader, iter_count: int):
        expected = {
            "models": [],
            "res": {
                ("res11", "res21"): [],
                ("res11", "res22"): [],
                ("res12", "res21"): [],
                ("res12", "res22"): []
            },
            "trials": []
        }
        for _ in range(iter_count):
            test_model = EmptyTestModel(time.time())
            sl.save_model(test_model)
            expected["models"].append(test_model)
            res = {("res11", "res21"): random.randint(-1e2, 1e2),
                   ("res11", "res22"): random.randint(-1e2, 1e2),
                   ("res12", "res21"): random.randint(-1e2, 1e2),
                   ("res12", "res22"): random.randint(-1e2, 1e2)}
            sl.save_res(res)
            expected["res"] = {key: [*value, res[key]] for key, value in expected["res"].items()}
            trial = {"tr1": [random.randint(-1e2, 1e2), random.randint(-1e2, 1e2)], 
                     "tr2": [random.randint(-1e2, 1e2), random.randint(-1e2, 1e2)]}
            sl.save_trial(trial)
            expected["trials"].append(trial)
            sl.next_iter()
        return expected

class TestOptunaLoader:
    def test_path_manager(self):
        clean_up("tests/Test/")
        manager = OptunaPathManager("tests/Test/")
        assert os.path.exists("tests/Test/studies/")
        assert manager.one_study(0) == "tests/Test/studies/study_0.joblib"
        clean_up("tests/Test/")
        
    def test_saveloader(self):
        clean_up("tests/Test/")
        sl = OptunaSaveLoaderFactory().create("tests/Test/")
        iter_count = 5
        for iter_id in range(iter_count):
            test_model = EmptyTestModel(time.time())
            sl.save_study(test_model)
            sl.next_iter()
            assert sl.load_study(iter_id).number == test_model.number
        clean_up("tests/Test/")
        