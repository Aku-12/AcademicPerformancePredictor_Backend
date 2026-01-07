import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data.preprocessor import DataPreprocessor
from data.loader import DataLoader
from models.academic_predictor import AcademicPredictor, check_gpu
from config import Config


class ModelTrainer:
    def __init__(self, model_type='xgboost_gpu', use_gpu=True):
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.preprocessor = DataPreprocessor()
        self.model = AcademicPredictor(model_type=model_type, use_gpu=use_gpu)
        self.loader = DataLoader()
        self.training_history = []

        # Print GPU info
        gpu_info = check_gpu()
        print("\n" + "=" * 60)
        print("GPU STATUS")
        print("=" * 60)
        if gpu_info['cuda_available']:
            print(f"CUDA Available: Yes")
            print(f"GPU Device: {gpu_info['device_name']}")
            print(f"GPU Count: {gpu_info['device_count']}")
        else:
            print("CUDA Available: No - Running on CPU")
        print("=" * 60 + "\n")

    def train_from_csv(self, filename='cs_academic_performance_1M.csv', sample_size=None):
        """Train model from CSV file with GPU acceleration."""
        print("=" * 60)
        print("ACADEMIC PERFORMANCE PREDICTOR - GPU TRAINING")
        print("=" * 60)

        # Load data
        df = self.loader.load_raw_data(filename)

        # Sample if needed
        if sample_size and len(df) > sample_size:
            print(f"Sampling {sample_size} records from {len(df)} total...")
            df = self.loader.sample_data(df, n_samples=sample_size)

        return self._train(df)

    def train_from_splits(self):
        """Train model from existing train/val/test splits."""
        train, val, test = self.loader.load_splits()
        return self._train(train, val_df=val, test_df=test)

    def _train(self, df, val_df=None, test_df=None):
        """Internal training method with GPU support."""
        print(f"\nPreprocessing {len(df)} records...")

        # Create splits if not provided
        if val_df is None or test_df is None:
            train_df, val_df, test_df = self.loader.create_train_test_split(df, save=True)
        else:
            train_df = df

        # Fit preprocessor on training data
        print("Fitting preprocessor...")
        self.preprocessor.fit(train_df)

        # Transform all sets
        train_processed = self.preprocessor.transform(train_df)
        val_processed = self.preprocessor.transform(val_df)
        test_processed = self.preprocessor.transform(test_df)

        # Prepare features and targets
        X_train = self.preprocessor.prepare_features(train_processed)
        y_train = self.preprocessor.prepare_target(train_df)

        X_val = self.preprocessor.prepare_features(val_processed)
        y_val = self.preprocessor.prepare_target(val_df)

        X_test = self.preprocessor.prepare_features(test_processed)
        y_test = self.preprocessor.prepare_target(test_df)

        if y_train is None:
            raise ValueError("Target column 'semester_gpa' not found in data")

        # Train model
        print(f"\nTraining {self.model_type} model...")
        feature_names = self.preprocessor.get_feature_names()
        metrics = self.model.train(X_train, y_train, feature_names=feature_names)

        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_pred = self.model.predict(X_val)
        metrics['val_mse'] = float(mean_squared_error(y_val, val_pred))
        metrics['val_mae'] = float(mean_absolute_error(y_val, val_pred))
        metrics['val_r2'] = float(r2_score(y_val, val_pred))
        metrics['val_rmse'] = float(mean_squared_error(y_val, val_pred, squared=False))

        # Evaluate on test set
        print("Evaluating on test set...")
        test_pred = self.model.predict(X_test)
        metrics['test_mse'] = float(mean_squared_error(y_test, test_pred))
        metrics['test_mae'] = float(mean_absolute_error(y_test, test_pred))
        metrics['test_r2'] = float(r2_score(y_test, test_pred))
        metrics['test_rmse'] = float(mean_squared_error(y_test, test_pred, squared=False))

        # Add training timestamp
        metrics['trained_at'] = datetime.now().isoformat()

        # Print results
        self._print_results(metrics, len(train_df))

        self.training_history.append(metrics)
        return metrics

    def _print_results(self, metrics, train_size):
        """Print training results."""
        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"Model Type: {metrics['model_type']}")
        print(f"Device: {metrics.get('device', 'cpu').upper()}")
        print(f"Training Samples: {train_size:,}")
        print(f"Training Time: {metrics.get('training_time_seconds', 'N/A')}s")

        print(f"\nCross-Validation R²: {metrics['cv_mean_r2']:.4f} ± {metrics['cv_std_r2']:.4f}")

        print(f"\nValidation Set:")
        print(f"  - R²:   {metrics['val_r2']:.4f}")
        print(f"  - RMSE: {metrics['val_rmse']:.4f}")
        print(f"  - MAE:  {metrics['val_mae']:.4f}")

        print(f"\nTest Set:")
        print(f"  - R²:   {metrics['test_r2']:.4f}")
        print(f"  - RMSE: {metrics['test_rmse']:.4f}")
        print(f"  - MAE:  {metrics['test_mae']:.4f}")
        print("=" * 60)

    def save_model(self):
        """Save trained model and preprocessor."""
        print(f"\nSaving model to {Config.MODEL_PATH}...")

        # Save model
        model_path = self.model.save(Config.MODEL_PATH)

        # Save preprocessor
        preprocessor_path = self.preprocessor.save(Config.MODEL_PATH)

        # Save training metrics
        metrics_path = os.path.join(Config.MODEL_PATH, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        paths = {
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'metrics_path': metrics_path
        }

        print("Model saved successfully!")
        return paths

    def get_feature_importance(self, top_n=10):
        """Get top N feature importance from trained model."""
        return self.model.get_top_features(n=top_n, feature_names=self.preprocessor.get_feature_names())


def main():
    """Main training script with GPU support."""
    import argparse

    parser = argparse.ArgumentParser(description='Train Academic Performance Predictor with GPU')
    parser.add_argument('--model', type=str, default='xgboost_gpu',
                        choices=['xgboost_gpu', 'xgboost_cpu', 'lightgbm_gpu', 'lightgbm_cpu',
                                 'random_forest', 'gradient_boosting', 'neural_network', 'ridge'],
                        help='Model type to train')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to use (None for all 1M)')
    parser.add_argument('--data', type=str, default='cs_academic_performance_1M.csv',
                        help='Dataset filename')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU even if available')

    args = parser.parse_args()

    use_gpu = not args.no_gpu

    # Train model
    trainer = ModelTrainer(model_type=args.model, use_gpu=use_gpu)
    metrics = trainer.train_from_csv(filename=args.data, sample_size=args.samples)

    # Save model
    paths = trainer.save_model()
    print(f"\nModel artifacts saved to: {Config.MODEL_PATH}")

    # Show feature importance
    print("\nTop 10 Feature Importance:")
    importance = trainer.get_feature_importance(top_n=10)
    if importance:
        for i, (feature, score) in enumerate(importance.items(), 1):
            print(f"  {i}. {feature}: {score:.4f}")


if __name__ == '__main__':
    main()
