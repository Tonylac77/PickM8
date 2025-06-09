import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

class Model:
    """Example custom model using a neural network"""
    
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
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
        # Use entropy as uncertainty measure
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        return entropy