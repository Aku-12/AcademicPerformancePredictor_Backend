from abc import ABC, abstractmethod
import joblib
import os

class BaseModel(ABC):
    def __init__(self, model_name='base_model'):
        self.model_name = model_name
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X, y):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass

    def save(self, path):
        """Save the model to disk."""
        if not os.path.exists(path):
            os.makedirs(path)

        model_path = os.path.join(path, f'{self.model_name}.joblib')
        joblib.dump({
            'model': self.model,
            'is_trained': self.is_trained,
            'model_name': self.model_name
        }, model_path)
        return model_path

    def load(self, path):
        """Load the model from disk."""
        model_path = os.path.join(path, f'{self.model_name}.joblib')
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            self.model = data['model']
            self.is_trained = data['is_trained']
            self.model_name = data['model_name']
            return True
        return False
