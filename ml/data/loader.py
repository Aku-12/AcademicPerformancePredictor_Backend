import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, 'raw')
        self.processed_path = os.path.join(base_path, 'processed')
        self.splits_path = os.path.join(base_path, 'splits')

    def load_raw_data(self, filename='cs_academic_performance_1M.csv'):
        """Load raw data from CSV file."""
        filepath = os.path.join(self.raw_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df

    def load_csv(self, filename, folder='raw'):
        """Load data from CSV file in specified folder."""
        if folder == 'raw':
            filepath = os.path.join(self.raw_path, filename)
        elif folder == 'processed':
            filepath = os.path.join(self.processed_path, filename)
        elif folder == 'splits':
            filepath = os.path.join(self.splits_path, filename)
        else:
            filepath = os.path.join(self.base_path, filename)

        return pd.read_csv(filepath)

    def load_from_dict(self, data_dict):
        """Load data from dictionary (API input)."""
        return pd.DataFrame([data_dict])

    def load_from_list(self, data_list):
        """Load data from list of dictionaries."""
        return pd.DataFrame(data_list)

    def create_train_test_split(self, df, test_size=0.2, val_size=0.1, random_state=42, save=True):
        """Split data into train, validation, and test sets."""
        # First split: train+val and test
        train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)

        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_ratio, random_state=random_state)

        print(f"Data split - Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")

        if save:
            self.save_splits(train, val, test)

        return train, val, test

    def save_splits(self, train, val, test):
        """Save train/val/test splits to CSV files."""
        os.makedirs(self.splits_path, exist_ok=True)

        train.to_csv(os.path.join(self.splits_path, 'train.csv'), index=False)
        val.to_csv(os.path.join(self.splits_path, 'val.csv'), index=False)
        test.to_csv(os.path.join(self.splits_path, 'test.csv'), index=False)

        print(f"Splits saved to {self.splits_path}")

    def load_splits(self):
        """Load existing train/val/test splits."""
        train = pd.read_csv(os.path.join(self.splits_path, 'train.csv'))
        val = pd.read_csv(os.path.join(self.splits_path, 'val.csv'))
        test = pd.read_csv(os.path.join(self.splits_path, 'test.csv'))

        return train, val, test

    def save_processed(self, df, filename='processed_data.csv'):
        """Save processed dataframe."""
        os.makedirs(self.processed_path, exist_ok=True)
        filepath = os.path.join(self.processed_path, filename)
        df.to_csv(filepath, index=False)
        return filepath

    def get_data_info(self, df):
        """Get information about the dataset."""
        info = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'target_stats': {}
        }

        if 'semester_gpa' in df.columns:
            info['target_stats'] = {
                'mean': float(df['semester_gpa'].mean()),
                'std': float(df['semester_gpa'].std()),
                'min': float(df['semester_gpa'].min()),
                'max': float(df['semester_gpa'].max())
            }

        return info

    def sample_data(self, df, n_samples=10000, random_state=42):
        """Sample a subset of data for quick testing."""
        if len(df) <= n_samples:
            return df
        return df.sample(n=n_samples, random_state=random_state)
