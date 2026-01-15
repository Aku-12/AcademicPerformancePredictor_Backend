import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_cohort_statistics(predictions_data: List[Dict]) -> Dict[str, Any]:
    """
    Calculate aggregate statistics for a cohort of students.

    Args:
        predictions_data: List of prediction dictionaries containing GPA and student data

    Returns:
        Dictionary with cohort-level statistics
    """
    if not predictions_data:
        return {}

    gpas = [p['predicted_gpa'] for p in predictions_data]

    # GPA statistics
    gpa_stats = {
        'mean': float(np.mean(gpas)),
        'median': float(np.median(gpas)),
        'std': float(np.std(gpas)),
        'min': float(np.min(gpas)),
        'max': float(np.max(gpas)),
        'q1': float(np.percentile(gpas, 25)),
        'q3': float(np.percentile(gpas, 75))
    }

    # Performance category distribution
    categories = [_get_gpa_category(gpa) for gpa in gpas]
    category_distribution = {
        'Distinction': categories.count('Distinction'),
        'First_Division': categories.count('First_Division'),
        'Second_Division': categories.count('Second_Division'),
        'Third_Division': categories.count('Third_Division'),
        'Pass': categories.count('Pass'),
        'Fail': categories.count('Fail')
    }

    # Risk level distribution
    risk_levels = []
    dropout_probs = []
    scholarship_eligible = 0

    for pred in predictions_data:
        if 'risk_assessment' in pred:
            risk_levels.append(pred['risk_assessment']['risk_level'])
            dropout_probs.append(pred['risk_assessment']['dropout_probability'])
        if pred.get('scholarship', {}).get('eligible', False):
            scholarship_eligible += 1

    risk_distribution = {}
    if risk_levels:
        risk_distribution = {
            'No_Risk': risk_levels.count('No_Risk'),
            'Low_Risk': risk_levels.count('Low_Risk'),
            'Medium_Risk': risk_levels.count('Medium_Risk'),
            'High_Risk': risk_levels.count('High_Risk')
        }

    return {
        'total_students': len(predictions_data),
        'gpa_statistics': gpa_stats,
        'category_distribution': category_distribution,
        'risk_distribution': risk_distribution,
        'average_dropout_probability': float(np.mean(dropout_probs)) if dropout_probs else 0,
        'scholarship_eligible_count': scholarship_eligible,
        'scholarship_eligible_percentage': round((scholarship_eligible / len(predictions_data)) * 100, 2)
    }


def calculate_feature_correlations(df: pd.DataFrame, target_column: str = 'semester_gpa') -> Dict[str, float]:
    """
    Calculate correlation between features and target variable.

    Args:
        df: DataFrame with features and target
        target_column: Name of target column

    Returns:
        Dictionary of feature correlations sorted by absolute value
    """
    if target_column not in df.columns:
        return {}

    # Calculate correlations
    correlations = df.corr()[target_column].drop(target_column)

    # Sort by absolute value
    correlations_abs = correlations.abs().sort_values(ascending=False)

    # Return as dictionary
    return {col: float(correlations[col]) for col in correlations_abs.index}


def analyze_demographic_performance(df: pd.DataFrame,
                                   demographic_col: str,
                                   target_col: str = 'semester_gpa') -> List[Dict]:
    """
    Analyze performance across different demographic groups.

    Args:
        df: DataFrame with student data
        demographic_col: Column name for demographic grouping
        target_col: Performance metric column

    Returns:
        List of dictionaries with group statistics
    """
    if demographic_col not in df.columns or target_col not in df.columns:
        return []

    results = []
    for group in df[demographic_col].unique():
        group_data = df[df[demographic_col] == group][target_col]

        results.append({
            'group': str(group),
            'count': int(len(group_data)),
            'mean_gpa': float(group_data.mean()),
            'median_gpa': float(group_data.median()),
            'std_gpa': float(group_data.std()),
            'pass_rate': float((group_data >= 2.0).sum() / len(group_data) * 100)
        })

    # Sort by mean GPA descending
    results.sort(key=lambda x: x['mean_gpa'], reverse=True)

    return results


def calculate_gpa_distribution(predictions: List[float], bins: int = 20) -> Dict[str, Any]:
    """
    Calculate GPA distribution for histogram visualization.

    Args:
        predictions: List of predicted GPAs
        bins: Number of histogram bins

    Returns:
        Dictionary with histogram data
    """
    hist, bin_edges = np.histogram(predictions, bins=bins, range=(0, 4.0))

    return {
        'bins': [{'min': float(bin_edges[i]), 'max': float(bin_edges[i+1]), 'count': int(hist[i])}
                 for i in range(len(hist))],
        'total': len(predictions)
    }


def identify_risk_factors_frequency(predictions_data: List[Dict]) -> List[Dict]:
    """
    Identify most common risk factors across all students.

    Args:
        predictions_data: List of prediction dictionaries with risk assessments

    Returns:
        List of risk factors with frequency counts
    """
    factor_counts = {}
    total_students = len(predictions_data)

    for pred in predictions_data:
        if 'risk_assessment' not in pred:
            continue

        risk_factors = pred['risk_assessment'].get('risk_factors', [])
        for factor in risk_factors:
            factor_name = factor['factor']
            severity = factor.get('severity', 'medium')

            if factor_name not in factor_counts:
                factor_counts[factor_name] = {
                    'factor': factor_name,
                    'count': 0,
                    'percentage': 0,
                    'severity': severity
                }

            factor_counts[factor_name]['count'] += 1

    # Calculate percentages
    for factor_name in factor_counts:
        factor_counts[factor_name]['percentage'] = round(
            (factor_counts[factor_name]['count'] / total_students) * 100, 2
        )

    # Sort by count descending
    result = list(factor_counts.values())
    result.sort(key=lambda x: x['count'], reverse=True)

    return result


def calculate_percentile_rank(gpa: float, all_gpas: List[float]) -> float:
    """
    Calculate percentile rank of a GPA among all GPAs.

    Args:
        gpa: Individual GPA
        all_gpas: List of all GPAs for comparison

    Returns:
        Percentile rank (0-100)
    """
    if not all_gpas:
        return 50.0

    rank = sum(1 for g in all_gpas if g < gpa)
    percentile = (rank / len(all_gpas)) * 100

    return round(percentile, 2)


def generate_what_if_scenarios(student_data: Dict,
                              preprocessor,
                              model,
                              scenarios: List[Dict]) -> List[Dict]:
    """
    Generate what-if scenario predictions.

    Args:
        student_data: Original student data
        preprocessor: Fitted preprocessor
        model: Trained model
        scenarios: List of {"field": "value"} modifications

    Returns:
        List of scenario results with predictions
    """
    results = []

    for scenario in scenarios:
        # Create modified data
        modified_data = student_data.copy()
        modified_data.update(scenario['changes'])

        # Make prediction
        try:
            df = pd.DataFrame([modified_data])
            X = preprocessor.transform(df)
            X_features = preprocessor.prepare_features(X)
            prediction = model.predict(X_features)[0]

            results.append({
                'scenario_name': scenario.get('name', 'Unnamed'),
                'changes': scenario['changes'],
                'predicted_gpa': float(prediction),
                'gpa_change': float(prediction - student_data.get('current_gpa', prediction)),
                'category': _get_gpa_category(prediction)
            })
        except Exception as e:
            results.append({
                'scenario_name': scenario.get('name', 'Unnamed'),
                'error': str(e)
            })

    return results


def calculate_support_services_demand(predictions_data: List[Dict]) -> List[Dict]:
    """
    Calculate demand for different support services.

    Args:
        predictions_data: List of predictions with interventions

    Returns:
        List of services with demand counts
    """
    service_counts = {}

    for pred in predictions_data:
        if 'interventions' not in pred:
            continue

        services = pred['interventions'].get('support_services', [])
        for service in services:
            if service not in service_counts:
                service_counts[service] = 0
            service_counts[service] += 1

    # Convert to list and sort
    result = [{'service': name, 'students_needing': count}
              for name, count in service_counts.items()]
    result.sort(key=lambda x: x['students_needing'], reverse=True)

    return result


def get_top_recommendations(predictions_data: List[Dict]) -> List[Dict]:
    """
    Get most frequently generated recommendations.

    Args:
        predictions_data: List of predictions with recommendations

    Returns:
        List of recommendations with frequency
    """
    recommendation_counts = {}

    for pred in predictions_data:
        recommendations = pred.get('recommendations', [])
        for rec in recommendations:
            if rec not in recommendation_counts:
                recommendation_counts[rec] = 0
            recommendation_counts[rec] += 1

    result = [{'recommendation': rec, 'frequency': count}
              for rec, count in recommendation_counts.items()]
    result.sort(key=lambda x: x['frequency'], reverse=True)

    return result[:20]  # Top 20


def calculate_scholarship_funnel(predictions_data: List[Dict]) -> Dict[str, Any]:
    """
    Calculate scholarship eligibility funnel.

    Args:
        predictions_data: List of predictions with scholarship data

    Returns:
        Funnel statistics
    """
    total = len(predictions_data)

    eligible = sum(1 for p in predictions_data
                   if p.get('scholarship', {}).get('eligible', False))

    # Calculate near-eligible (score >= 60)
    near_eligible = sum(1 for p in predictions_data
                       if not p.get('scholarship', {}).get('eligible', False)
                       and p.get('scholarship', {}).get('eligibility_score', 0) >= 60)

    # By type
    scholarship_types = {}
    for pred in predictions_data:
        if pred.get('scholarship', {}).get('eligible', False):
            stype = pred['scholarship'].get('scholarship_type', 'Unknown')
            scholarship_types[stype] = scholarship_types.get(stype, 0) + 1

    return {
        'total_students': total,
        'eligible': eligible,
        'eligible_percentage': round((eligible / total) * 100, 2) if total > 0 else 0,
        'near_eligible': near_eligible,
        'near_eligible_percentage': round((near_eligible / total) * 100, 2) if total > 0 else 0,
        'not_eligible': total - eligible - near_eligible,
        'scholarship_types': scholarship_types
    }


def compare_to_similar_students(student_data: Dict,
                               all_students: pd.DataFrame,
                               target_col: str = 'semester_gpa',
                               similarity_features: List[str] = None) -> Dict[str, Any]:
    """
    Compare a student to similar students.

    Args:
        student_data: Individual student data
        all_students: DataFrame with all students
        target_col: Performance metric column
        similarity_features: Features to use for similarity matching

    Returns:
        Comparison statistics
    """
    if similarity_features is None:
        similarity_features = ['program', 'institution_type', 'gender']

    # Find similar students
    mask = pd.Series([True] * len(all_students))
    for feature in similarity_features:
        if feature in student_data and feature in all_students.columns:
            mask &= (all_students[feature] == student_data[feature])

    similar_students = all_students[mask]

    if len(similar_students) == 0:
        return {'error': 'No similar students found'}

    return {
        'count': len(similar_students),
        'mean_gpa': float(similar_students[target_col].mean()),
        'median_gpa': float(similar_students[target_col].median()),
        'top_performers': {
            'count': int((similar_students[target_col] >= 3.6).sum()),
            'percentage': round((similar_students[target_col] >= 3.6).sum() / len(similar_students) * 100, 2)
        },
        'at_risk': {
            'count': int((similar_students[target_col] < 2.4).sum()),
            'percentage': round((similar_students[target_col] < 2.4).sum() / len(similar_students) * 100, 2)
        },
        'similarity_criteria': similarity_features
    }


def _get_gpa_category(gpa: float) -> str:
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


def calculate_behavioral_profile(student_data: Dict) -> Dict[str, float]:
    """
    Create behavioral profile for radar chart.

    Args:
        student_data: Student input data

    Returns:
        Dictionary with normalized scores (0-100) for each dimension
    """
    profile = {}

    # Study habits (0-100 scale)
    study_hours = student_data.get('daily_study_hours', 3)
    profile['Study_Habits'] = min((study_hours / 6) * 100, 100)

    # Attendance (already 0-100)
    profile['Attendance'] = student_data.get('attendance_percentage', 75)

    # Motivation (convert to 0-100)
    motivation_map = {
        'Very_Low': 10,
        'Low_Motivation': 30,
        'Moderately_Motivated': 50,
        'Motivated': 75,
        'Highly_Motivated': 95
    }
    motivation = student_data.get('motivation_level', 'Moderately_Motivated')
    profile['Motivation'] = motivation_map.get(motivation, 50)

    # Stress (inverted - lower stress is better)
    stress_map = {
        'Very_Low': 95,
        'Low': 80,
        'Moderate': 60,
        'High': 30,
        'Very_High': 10
    }
    stress = student_data.get('stress_level', 'Moderate')
    profile['Stress_Management'] = stress_map.get(stress, 60)

    # Class participation
    participation_map = {
        'Inactive': 10,
        'Passive': 30,
        'Moderate': 50,
        'Active': 75,
        'Very_Active': 95
    }
    participation = student_data.get('class_participation', 'Moderate')
    profile['Engagement'] = participation_map.get(participation, 50)

    # Assignment submission
    assignment_map = {
        'Rarely_Submits': 10,
        'Often_Late': 30,
        'Sometimes_Late': 50,
        'Usually_On_Time': 75,
        'Always_On_Time': 95
    }
    assignment = student_data.get('assignment_submission', 'Usually_On_Time')
    profile['Assignment_Quality'] = assignment_map.get(assignment, 75)

    return profile
