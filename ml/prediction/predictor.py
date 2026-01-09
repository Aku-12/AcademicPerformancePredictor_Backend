import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.academic_predictor import AcademicPredictor
from data.preprocessor import DataPreprocessor
from data.loader import DataLoader

class PredictionService:
    # Default values for missing fields
    DEFAULT_VALUES = {
        # Numerical defaults (median/typical values)
        'current_age': 20,
        'plus_two_gpa': 3.0,
        'family_monthly_income_npr': 50000,
        'attendance_percentage': 75,
        'daily_study_hours': 3,
        'internal_marks': 60,
        'external_marks': 60,
        # Categorical defaults (neutral/common values)
        'gender': 'Male',
        'institution_type': 'Private',
        'institution_tier': 'Tier_2',
        'program': 'BSc_CSIT',
        'school_type': 'Private',
        'plus_two_stream': 'Science',
        'plus_two_gpa_category': 'Second_Division',
        'family_income_category': '40000_60000',
        'father_education': 'Higher_Secondary',
        'mother_education': 'Higher_Secondary',
        'accommodation_type': 'With_Family',
        'attendance_category': 'Good_75_90',
        'daily_study_hours_category': '2_4_hours',
        'learning_style': 'Visual',
        'prior_programming_experience': 'Basic',
        'english_proficiency': 'Intermediate',
        'mathematics_aptitude': 'Average',
        'device_ownership': 'Personal_Laptop',
        'internet_access': 'Home_WiFi',
        'extracurricular_activities': 'None',
        'part_time_work': 'No',
        'class_participation': 'Moderate',
        'assignment_submission': 'Usually_On_Time',
        'motivation_level': 'Moderately_Motivated',
        'stress_level': 'Moderate',
        'career_clarity': 'Somewhat_Clear',
    }

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.loader = DataLoader()
        self._load_model()

    def _load_model(self):
        """Load trained model and preprocessor."""
        model_path = os.path.join(Config.MODEL_PATH, 'academic_predictor.joblib')
        preprocessor_path = os.path.join(Config.MODEL_PATH, 'preprocessor.joblib')

        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            self.model = AcademicPredictor()
            self.model.load(Config.MODEL_PATH)

            self.preprocessor = DataPreprocessor()
            self.preprocessor.load(Config.MODEL_PATH)

            print("Model and preprocessor loaded successfully!")
            return True

        print("Warning: Model not found. Please train the model first.")
        return False

    def reload_model(self):
        """Reload model from disk."""
        return self._load_model()

    def _fill_defaults(self, student_data):
        """Fill missing fields with default values."""
        filled_data = dict(self.DEFAULT_VALUES)  # Start with defaults
        filled_data.update(student_data)  # Override with provided values
        return filled_data

    def is_ready(self):
        """Check if model is loaded and ready."""
        return (self.model is not None and
                self.model.is_trained and
                self.preprocessor is not None and
                self.preprocessor.is_fitted)

    def predict_single(self, student_data):
        """Predict GPA for a single student."""
        if not self.is_ready():
            raise ValueError("Model not loaded. Please train the model first.")

        # Fill missing fields with defaults
        filled_data = self._fill_defaults(student_data)

        df = self.loader.load_from_dict(filled_data)
        df_processed = self.preprocessor.transform(df)
        X = self.preprocessor.prepare_features(df_processed)

        prediction = self.model.predict(X)[0]

        return {
            'predicted_gpa': round(float(prediction), 2),
            'gpa_category': self._get_gpa_category(prediction),
            'input_data': student_data  # Return original input, not filled
        }

    def predict_batch(self, students_data):
        """Predict GPA for multiple students."""
        if not self.is_ready():
            raise ValueError("Model not loaded. Please train the model first.")

        # Fill missing fields with defaults for each student
        filled_data = [self._fill_defaults(student) for student in students_data]

        df = self.loader.load_from_list(filled_data)
        df_processed = self.preprocessor.transform(df)
        X = self.preprocessor.prepare_features(df_processed)

        predictions = self.model.predict(X)

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'student_index': i,
                'predicted_gpa': round(float(pred), 2),
                'gpa_category': self._get_gpa_category(pred),
                'input_data': students_data[i]
            })

        return results

    def _get_gpa_category(self, gpa):
        """Categorize GPA based on Nepali grading system."""
        if gpa >= 3.6:
            return 'Distinction'
        elif gpa >= 3.2:
            return 'First_Division'
        elif gpa >= 2.8:
            return 'Second_Division'
        elif gpa >= 2.4:
            return 'Third_Division'
        elif gpa >= 2.0:
            return 'Pass'
        else:
            return 'Fail'

    def get_model_info(self):
        """Get information about the loaded model."""
        if not self.is_ready():
            return {'status': 'not_loaded'}

        return {
            'status': 'ready',
            'model_type': self.model.model_type,
            'metrics': self.model.metrics,
            'features': {
                'numerical': self.preprocessor.numerical_features,
                'categorical': self.preprocessor.categorical_features,
                'total': len(self.preprocessor.feature_columns)
            }
        }

    def get_feature_importance(self, top_n=10):
        """Get top feature importance."""
        if not self.is_ready():
            return None

        return self.model.get_top_features(
            n=top_n,
            feature_names=self.preprocessor.get_feature_names()
        )

    def get_required_features(self):
        """Get list of required input features for prediction."""
        return {
            'numerical': self.preprocessor.numerical_features,
            'categorical': self.preprocessor.categorical_features
        }
