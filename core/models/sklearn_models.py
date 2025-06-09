import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from .base import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def train(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def get_uncertainty(self, X):
        proba = self.predict_proba(X)
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        return entropy

class SVMModel(BaseModel):
    def __init__(self, kernel='rbf', C=1.0, random_state=42):
        self.model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
            random_state=random_state
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def train(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def get_uncertainty(self, X):
        proba = self.predict_proba(X)
        margin = 1 - np.max(proba, axis=1)
        return margin

class EnsembleModel(BaseModel):
    def __init__(self, n_members=3, base_model='random_forest', random_state=42):
        self.n_members = n_members
        self.base_model = base_model
        self.random_state = random_state
        self.members = []
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def _create_member(self, seed):
        if self.base_model == 'random_forest':
            return RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
        elif self.base_model == 'svm':
            return SVC(probability=True, random_state=seed)
        else:
            raise ValueError(f"Unknown base model: {self.base_model}")
    
    def train(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)
        self.members = []
        
        n_samples = len(X)
        for i in range(self.n_members):
            seed = self.random_state + i
            np.random.seed(seed)
            
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y_encoded[indices]
            
            member = self._create_member(seed)
            member.fit(X_bootstrap, y_bootstrap)
            self.members.append(member)
        
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        for member in self.members:
            pred = member.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 0, predictions
        )
        
        return self.label_encoder.inverse_transform(ensemble_pred)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        all_proba = []
        for member in self.members:
            proba = member.predict_proba(X)
            all_proba.append(proba)
        
        return np.mean(all_proba, axis=0)
    
    def get_uncertainty(self, X):
        all_proba = []
        for member in self.members:
            proba = member.predict_proba(X)
            all_proba.append(proba)
        
        all_proba = np.array(all_proba)
        variance = np.mean(np.var(all_proba, axis=0), axis=1)
        return variance