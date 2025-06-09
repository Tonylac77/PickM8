from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import importlib.util
import pickle

class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        pass
    
    @abstractmethod
    def get_uncertainty(self, X):
        pass
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

class ModelLoader:
    @staticmethod
    def load_builtin_model(model_type, **kwargs):
        from .sklearn_models import RandomForestModel, SVMModel, EnsembleModel
        
        models = {
            'random_forest': RandomForestModel,
            'svm': SVMModel,
            'ensemble': EnsembleModel
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type](**kwargs)
    
    @staticmethod
    def load_custom_model(model_path):
        spec = importlib.util.spec_from_file_location("custom_model", model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'Model'):
            return module.Model()
        else:
            raise AttributeError("Custom model file must define a 'Model' class")