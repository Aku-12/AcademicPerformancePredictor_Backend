from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np

from config import Config
from prediction.predictor import PredictionService
from training.trainer import ModelTrainer
from models.academic_predictor import check_gpu
from utils.helpers import validate_student_data, format_prediction_response, get_performance_summary
from utils.insights import (
    calculate_cohort_statistics,
    calculate_feature_correlations,
    analyze_demographic_performance,
    calculate_gpa_distribution,
    identify_risk_factors_frequency,
    calculate_percentile_rank,
    generate_what_if_scenarios,
    calculate_support_services_demand,
    get_top_recommendations,
    calculate_scholarship_funnel,
    compare_to_similar_students,
    calculate_behavioral_profile
)

app = Flask(__name__)
CORS(app)

# Initialize prediction service
prediction_service = PredictionService()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_ready': prediction_service.is_ready()
    })


@app.route('/gpu/status', methods=['GET'])
def gpu_status():
    """Get GPU status."""
    gpu_info = check_gpu()
    return jsonify({
        'cuda_available': gpu_info['cuda_available'],
        'device_count': gpu_info['device_count'],
        'device_name': gpu_info['device_name'],
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    return jsonify(prediction_service.get_model_info())


@app.route('/model/features', methods=['GET'])
def model_features():
    """Get required features for prediction."""
    if not prediction_service.is_ready():
        return jsonify({'error': 'Model not trained'}), 400
    return jsonify(prediction_service.get_required_features())


@app.route('/predict', methods=['POST'])
def predict():
    """Predict GPA for a single student."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Make prediction
        prediction = prediction_service.predict_single(data)
        response = format_prediction_response(prediction)

        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict GPA for multiple students."""
    try:
        data = request.get_json()

        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected a list of student data'}), 400

        predictions = prediction_service.predict_batch(data)
        results = [format_prediction_response(p) for p in predictions]

        return jsonify({'predictions': results, 'count': len(results)})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Batch prediction failed', 'details': str(e)}), 500


@app.route('/predict/summary', methods=['POST'])
def predict_with_summary():
    """Predict GPA and return comprehensive summary."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        prediction = prediction_service.predict_single(data)
        summary = get_performance_summary(prediction['predicted_gpa'], data)

        return jsonify(summary)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


@app.route('/train', methods=['POST'])
def train_model():
    """Train the model with GPU acceleration."""
    try:
        data = request.get_json() or {}

        model_type = data.get('model_type', 'xgboost_gpu')
        sample_size = data.get('sample_size', None)
        use_gpu = data.get('use_gpu', True)

        # Validate model type
        valid_models = [
            'xgboost_gpu', 'xgboost_cpu',
            'lightgbm_gpu', 'lightgbm_cpu',
            'random_forest', 'gradient_boosting',
            'neural_network', 'ridge'
        ]

        if model_type not in valid_models:
            return jsonify({
                'error': f'Invalid model type. Choose from: {valid_models}'
            }), 400

        trainer = ModelTrainer(model_type=model_type, use_gpu=use_gpu)

        # Train with CSV data
        metrics = trainer.train_from_csv(
            filename='cs_academic_performance_1M.csv',
            sample_size=sample_size
        )
        paths = trainer.save_model()

        # Reload prediction service
        global prediction_service
        prediction_service = PredictionService()

        return jsonify({
            'message': 'Model trained successfully',
            'metrics': metrics,
            'paths': paths,
            'gpu_used': trainer.use_gpu and torch.cuda.is_available()
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Training failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/features/importance', methods=['GET'])
def feature_importance():
    """Get feature importance."""
    try:
        if not prediction_service.is_ready():
            return jsonify({'error': 'Model not trained'}), 400

        top_n = request.args.get('top', 10, type=int)
        importance = prediction_service.get_feature_importance(top_n=top_n)

        return jsonify({'feature_importance': importance})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reload', methods=['POST'])
def reload_model():
    """Reload model from disk."""
    try:
        success = prediction_service.reload_model()
        if success:
            return jsonify({'message': 'Model reloaded successfully'})
        return jsonify({'error': 'Failed to reload model'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# INSIGHTS ENDPOINTS
# ============================================================================

@app.route('/insights/student', methods=['POST'])
def student_insights():
    """Get comprehensive insights for a single student."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Make prediction with full summary
        prediction = prediction_service.predict_single(data)
        summary = get_performance_summary(prediction['predicted_gpa'], data)

        # Add behavioral profile
        behavioral_profile = calculate_behavioral_profile(data)

        # Load cohort data for percentile calculation
        cohort_gpas = _load_cohort_gpas()
        percentile = calculate_percentile_rank(prediction['predicted_gpa'], cohort_gpas)

        # Generate what-if scenarios
        default_scenarios = [
            {
                'name': 'Improve Attendance to 85%',
                'changes': {'attendance_percentage': 85}
            },
            {
                'name': 'Increase Study Hours to 5 hrs/day',
                'changes': {'daily_study_hours': 5}
            },
            {
                'name': 'Perfect Attendance + More Study',
                'changes': {'attendance_percentage': 95, 'daily_study_hours': 6}
            },
            {
                'name': 'Improve Assignment Submission',
                'changes': {'assignment_submission': 'Always_On_Time'}
            }
        ]

        what_if_results = generate_what_if_scenarios(
            data,
            prediction_service.preprocessor,
            prediction_service.model,
            default_scenarios
        )

        # Compare to similar students
        similar_comparison = _compare_to_similar(data)

        return jsonify({
            'prediction': summary,
            'behavioral_profile': behavioral_profile,
            'percentile_rank': percentile,
            'what_if_scenarios': what_if_results,
            'similar_students': similar_comparison,
            'cohort_size': len(cohort_gpas)
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Failed to generate insights',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/insights/cohort', methods=['GET'])
def cohort_insights():
    """Get aggregate insights for the entire cohort."""
    try:
        # Load training/test data
        df = _load_cohort_data()

        if df.empty:
            return jsonify({'error': 'No cohort data available'}), 404

        # Make predictions for sample if needed (reduced for performance)
        sample_size = min(500, len(df))  # Reduced from 1000 to 500
        df_sample = df.sample(n=sample_size, random_state=42)

        predictions_data = []
        for _, row in df_sample.iterrows():
            row_dict = row.to_dict()
            try:
                pred = prediction_service.predict_single(row_dict)
                summary = get_performance_summary(pred['predicted_gpa'], row_dict)
                predictions_data.append(summary)
            except:
                continue

        # Calculate statistics
        cohort_stats = calculate_cohort_statistics(predictions_data)

        # GPA distribution
        gpas = [p['predicted_gpa'] for p in predictions_data]
        gpa_distribution = calculate_gpa_distribution(gpas)

        # Risk factors frequency
        risk_factors = identify_risk_factors_frequency(predictions_data)

        # Support services demand
        services_demand = calculate_support_services_demand(predictions_data)

        # Top recommendations
        top_recommendations = get_top_recommendations(predictions_data)

        # Scholarship funnel
        scholarship_funnel = calculate_scholarship_funnel(predictions_data)

        return jsonify({
            'statistics': cohort_stats,
            'gpa_distribution': gpa_distribution,
            'risk_factors': risk_factors[:10],  # Top 10
            'support_services_demand': services_demand,
            'top_recommendations': top_recommendations[:15],
            'scholarship_funnel': scholarship_funnel,
            'sample_size': sample_size
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Failed to generate cohort insights',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/insights/correlations', methods=['GET'])
def feature_correlations():
    """Get feature correlations with GPA."""
    try:
        df = _load_cohort_data()

        if df.empty:
            return jsonify({'error': 'No data available'}), 404

        # Calculate correlations
        correlations = calculate_feature_correlations(df, 'semester_gpa')

        # Get top positive and negative correlations
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            'all_correlations': correlations,
            'top_positive': dict(sorted_corr[:15]),
            'top_negative': dict(sorted_corr[-15:]),
            'total_features': len(correlations)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/insights/demographics/<demographic_type>', methods=['GET'])
def demographic_insights(demographic_type):
    """Get performance insights by demographic category."""
    try:
        df = _load_cohort_data()

        if df.empty:
            return jsonify({'error': 'No data available'}), 404

        # Map demographic type to column name
        demographic_columns = {
            'gender': 'gender',
            'institution': 'institution_type',
            'program': 'program',
            'school': 'school_type',
            'income': 'family_income_category',
            'accommodation': 'accommodation_type'
        }

        if demographic_type not in demographic_columns:
            return jsonify({
                'error': f'Invalid demographic type. Choose from: {list(demographic_columns.keys())}'
            }), 400

        col_name = demographic_columns[demographic_type]

        if col_name not in df.columns:
            return jsonify({'error': f'Column {col_name} not found in data'}), 404

        # Analyze performance
        analysis = analyze_demographic_performance(df, col_name, 'semester_gpa')

        return jsonify({
            'demographic_type': demographic_type,
            'groups': analysis
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/insights/what-if', methods=['POST'])
def what_if_analysis():
    """Perform what-if scenario analysis."""
    try:
        data = request.get_json()

        if not data or 'student_data' not in data or 'scenarios' not in data:
            return jsonify({
                'error': 'Provide student_data and scenarios'
            }), 400

        student_data = data['student_data']
        scenarios = data['scenarios']

        # Generate scenarios
        results = generate_what_if_scenarios(
            student_data,
            prediction_service.preprocessor,
            prediction_service.model,
            scenarios
        )

        return jsonify({
            'original_data': student_data,
            'scenarios': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/insights/behavioral-profile', methods=['POST'])
def behavioral_profile():
    """Get behavioral profile for radar chart."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        profile = calculate_behavioral_profile(data)

        # Also get ideal profile for comparison
        ideal_profile = {
            'Study_Habits': 85,
            'Attendance': 90,
            'Motivation': 90,
            'Stress_Management': 80,
            'Engagement': 85,
            'Assignment_Quality': 90
        }

        return jsonify({
            'student_profile': profile,
            'ideal_profile': ideal_profile,
            'dimensions': list(profile.keys())
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# HELPER FUNCTIONS FOR INSIGHTS
# ============================================================================

def _load_cohort_data() -> pd.DataFrame:
    """Load cohort data from CSV files."""
    try:
        import os
        data_dir = os.path.join(os.path.dirname(__file__), 'data', 'splits')

        # Try to load test data first, then validation
        for filename in ['test.csv', 'val.csv', 'train.csv']:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                return df

        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading cohort data: {e}")
        return pd.DataFrame()


def _load_cohort_gpas() -> list:
    """Load all GPAs from cohort for percentile calculation."""
    df = _load_cohort_data()
    if not df.empty and 'semester_gpa' in df.columns:
        return df['semester_gpa'].dropna().tolist()
    return []


def _compare_to_similar(student_data: dict) -> dict:
    """Compare student to similar students in cohort."""
    df = _load_cohort_data()

    if df.empty:
        return {'error': 'No cohort data available'}

    similarity_features = ['program', 'institution_type', 'gender']

    return compare_to_similar_students(
        student_data,
        df,
        'semester_gpa',
        similarity_features
    )


if __name__ == '__main__':
    gpu_info = check_gpu()

    print("=" * 60)
    print("CS ACADEMIC PERFORMANCE PREDICTOR - ML SERVICE")
    print("=" * 60)
    print(f"GPU Available: {gpu_info['cuda_available']}")
    if gpu_info['cuda_available']:
        print(f"GPU Device: {gpu_info['device_name']}")
    print(f"Starting server on port {Config.ML_PORT}...")
    print(f"Model ready: {prediction_service.is_ready()}")
    print("=" * 60)

    app.run(host='0.0.0.0', port=Config.ML_PORT, debug=True)
