import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

        # Define feature columns based on cs_academic_performance dataset
        self.numerical_features = [
            'current_age',
            'plus_two_gpa',
            'family_monthly_income_npr',
            'attendance_percentage',
            'daily_study_hours',
            'internal_marks',
            'external_marks'
        ]

        self.categorical_features = [
            'gender',
            'institution_type',
            'institution_tier',
            'program',
            'school_type',
            'plus_two_stream',
            'plus_two_gpa_category',
            'family_income_category',
            'father_education',
            'mother_education',
            'accommodation_type',
            'attendance_category',
            'daily_study_hours_category',
            'learning_style',
            'prior_programming_experience',
            'english_proficiency',
            'mathematics_aptitude',
            'device_ownership',
            'internet_access',
            'extracurricular_activities',
            'part_time_work',
            'class_participation',
            'assignment_submission',
            'motivation_level',
            'stress_level',
            'career_clarity'
        ]

        self.target_column = 'semester_gpa'
        self.feature_columns = self.numerical_features + self.categorical_features
        self.is_fitted = False

    def fit(self, df):
        """Fit the preprocessor on training data."""
        df_copy = df.copy()

        # Encode categorical columns
        for col in self.categorical_features:
            if col in df_copy.columns:
                self.label_encoders[col] = LabelEncoder()
                # Handle missing values
                df_copy[col] = df_copy[col].fillna('Unknown')
                self.label_encoders[col].fit(df_copy[col].astype(str))

        # Fit scaler on numerical features
        available_numerical = [col for col in self.numerical_features if col in df_copy.columns]
        if available_numerical:
            # Fill missing numerical values with median
            for col in available_numerical:
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            self.scaler.fit(df_copy[available_numerical])

        self.is_fitted = True
        return self

    def transform(self, df):
        """Transform the data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        df_copy = df.copy()

        # Encode categorical columns
        for col in self.categorical_features:
            if col in df_copy.columns and col in self.label_encoders:
                df_copy[col] = df_copy[col].fillna('Unknown')
                # Handle unseen labels
                le = self.label_encoders[col]
                df_copy[col] = df_copy[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Scale numerical features
        available_numerical = [col for col in self.numerical_features if col in df_copy.columns]
        if available_numerical:
            for col in available_numerical:
                df_copy[col] = df_copy[col].fillna(df_copy[col].median() if len(df_copy) > 1 else 0)
            df_copy[available_numerical] = self.scaler.transform(df_copy[available_numerical])

        return df_copy

    def fit_transform(self, df):
        """Fit and transform the data."""
        self.fit(df)
        return self.transform(df)

    def prepare_features(self, df):
        """Prepare feature matrix for model training/prediction."""
        available_features = [col for col in self.feature_columns if col in df.columns]
        return df[available_features].values

    def prepare_target(self, df):
        """Prepare target variable."""
        if self.target_column in df.columns:
            return df[self.target_column].values
        return None

    def get_feature_names(self):
        """Get list of feature names."""
        return self.feature_columns

    def save(self, path):
        """Save preprocessor to disk."""
        if not os.path.exists(path):
            os.makedirs(path)

        preprocessor_path = os.path.join(path, 'preprocessor.joblib')
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'is_fitted': self.is_fitted
        }, preprocessor_path)
        return preprocessor_path

    def load(self, path):
        """Load preprocessor from disk."""
        preprocessor_path = os.path.join(path, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            data = joblib.load(preprocessor_path)
            self.scaler = data['scaler']
            self.label_encoders = data['label_encoders']
            self.numerical_features = data['numerical_features']
            self.categorical_features = data['categorical_features']
            self.feature_columns = data['feature_columns']
            self.target_column = data['target_column']
            self.is_fitted = data['is_fitted']
            return True
        return False
