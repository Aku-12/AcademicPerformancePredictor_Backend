import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from .base_model import BaseModel

# Check GPU availability
def check_gpu():
    """Check if GPU is available."""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    return gpu_info


class AcademicPredictor(BaseModel):
    def __init__(self, model_type='xgboost_gpu', use_gpu=True):
        super().__init__(model_name='academic_predictor')
        self.model_type = model_type
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.model = self._create_model(model_type)
        self.metrics = {}
        self.feature_names = []

        if self.use_gpu:
            print(f"GPU enabled: {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU")

    def _create_model(self, model_type):
        """Create the specified model type with GPU support where available."""
        try:
            import xgboost as xgb
            import lightgbm as lgb
        except ImportError:
            xgb = None
            lgb = None

        models = {}

        # XGBoost with GPU
        if xgb:
            models['xgboost_gpu'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                tree_method='hist',  # Use 'gpu_hist' for GPU
                device='cuda' if self.use_gpu else 'cpu',
                random_state=42,
                n_jobs=-1
            )
            models['xgboost_cpu'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                tree_method='hist',
                device='cpu',
                random_state=42,
                n_jobs=-1
            )

        # LightGBM with GPU
        if lgb:
            models['lightgbm_gpu'] = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                device='gpu' if self.use_gpu else 'cpu',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            models['lightgbm_cpu'] = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                device='cpu',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

        # CPU-only models
        models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        models['ridge'] = Ridge(alpha=1.0)

        # Neural Network (will be created separately)
        if model_type == 'neural_network':
            return None  # Created during training

        return models.get(model_type, models.get('xgboost_gpu') or models['random_forest'])

    def train(self, X, y, feature_names=None):
        """Train the model with GPU acceleration."""
        if feature_names:
            self.feature_names = feature_names

        print(f"Training {self.model_type} model on {self.device.upper()}...")
        print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

        if self.model_type == 'neural_network':
            return self._train_neural_network(X, y)

        # Standard sklearn/xgboost/lightgbm training
        from tqdm import tqdm
        import time

        start_time = time.time()

        # Cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2', n_jobs=-1)
        print(f"CV R² scores: {cv_scores}")

        # Full training
        print("Training on full dataset...")
        self.model.fit(X, y)
        self.is_trained = True

        training_time = time.time() - start_time

        self.metrics = {
            'cv_mean_r2': float(np.mean(cv_scores)),
            'cv_std_r2': float(np.std(cv_scores)),
            'model_type': self.model_type,
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'training_time_seconds': round(training_time, 2),
            'device': self.device
        }

        print(f"Training completed in {training_time:.2f}s")
        print(f"Mean CV R²: {self.metrics['cv_mean_r2']:.4f}")

        return self.metrics

    def _train_neural_network(self, X, y):
        """Train PyTorch neural network on GPU."""
        from torch.utils.data import DataLoader, TensorDataset
        import time

        start_time = time.time()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Create model
        self.model = NeuralNetworkRegressor(
            input_dim=X.shape[1],
            hidden_dims=[256, 128, 64],
            dropout=0.3
        ).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

        # Training loop
        epochs = 100
        best_loss = float('inf')
        patience_counter = 0

        print(f"Training Neural Network on {self.device.upper()}...")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.is_trained = True
        training_time = time.time() - start_time

        # Calculate R² on training data
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze().cpu().numpy()
            from sklearn.metrics import r2_score
            r2 = r2_score(y, predictions)

        self.metrics = {
            'cv_mean_r2': float(r2),
            'cv_std_r2': 0.0,
            'model_type': 'neural_network',
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'training_time_seconds': round(training_time, 2),
            'device': self.device,
            'final_loss': best_loss
        }

        print(f"Training completed in {training_time:.2f}s")
        print(f"R²: {r2:.4f}")

        return self.metrics

    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if self.model_type == 'neural_network':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                predictions = self.model(X_tensor).squeeze().cpu().numpy()
        else:
            predictions = self.model.predict(X)

        # Clip to valid GPA range
        predictions = np.clip(predictions, 0, 4.0)
        return predictions

    def get_feature_importance(self, feature_names=None):
        """Get feature importance."""
        if not self.is_trained:
            return None

        names = feature_names or self.feature_names

        # XGBoost/LightGBM
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if names and len(names) == len(importances):
                importance_dict = dict(zip(names, importances.tolist()))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return importances.tolist()

        return None

    def get_top_features(self, n=10, feature_names=None):
        """Get top N most important features."""
        importance = self.get_feature_importance(feature_names)
        if importance and isinstance(importance, dict):
            return dict(list(importance.items())[:n])
        return None


class NeuralNetworkRegressor(nn.Module):
    """PyTorch Neural Network for GPA prediction."""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(NeuralNetworkRegressor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
