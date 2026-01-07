def validate_student_data(data):
    """Validate student input data for CS Academic Performance prediction."""
    required_numerical = [
        'current_age',
        'attendance_percentage',
        'daily_study_hours'
    ]

    optional_numerical = [
        'plus_two_gpa',
        'family_monthly_income_npr',
        'internal_marks',
        'external_marks'
    ]

    categorical_fields = [
        'gender',
        'institution_type',
        'program',
        'attendance_category',
        'stress_level',
        'motivation_level',
        'class_participation',
        'assignment_submission'
    ]

    errors = []

    # Check required numerical fields
    for field in required_numerical:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Validate numerical ranges
    validations = {
        'current_age': (18, 25),
        'plus_two_gpa': (0, 4.0),
        'attendance_percentage': (0, 100),
        'daily_study_hours': (0, 24),
        'internal_marks': (0, 100),
        'external_marks': (0, 100),
        'family_monthly_income_npr': (0, 1000000)
    }

    for field, (min_val, max_val) in validations.items():
        if field in data and data[field] is not None:
            value = data[field]
            if not isinstance(value, (int, float)):
                errors.append(f"{field} must be a number")
            elif value < min_val or value > max_val:
                errors.append(f"{field} must be between {min_val} and {max_val}")

    # Validate categorical fields
    valid_values = {
        'gender': ['Male', 'Female'],
        'stress_level': ['Very_Low', 'Low', 'Moderate', 'High', 'Very_High'],
        'motivation_level': ['Very_Low', 'Low_Motivation', 'Moderately_Motivated', 'Motivated', 'Highly_Motivated'],
        'attendance_category': ['Poor_Below_60', 'Average_60_75', 'Good_75_90', 'Excellent_Above_90'],
        'class_participation': ['Inactive', 'Passive', 'Moderate', 'Active', 'Very_Active'],
        'assignment_submission': ['Rarely_Submits', 'Often_Late', 'Sometimes_Late', 'Usually_On_Time', 'Always_On_Time']
    }

    for field, valid in valid_values.items():
        if field in data and data[field] is not None:
            if data[field] not in valid:
                errors.append(f"{field} must be one of: {', '.join(valid)}")

    return {
        'is_valid': len(errors) == 0,
        'errors': errors
    }


def format_prediction_response(prediction, include_recommendations=True):
    """Format prediction response with optional recommendations."""
    response = {
        'predicted_gpa': prediction['predicted_gpa'],
        'gpa_category': prediction.get('gpa_category', _get_gpa_category(prediction['predicted_gpa']))
    }

    if include_recommendations:
        response['recommendations'] = _generate_recommendations(
            prediction['predicted_gpa'],
            prediction.get('input_data', {})
        )

    return response


def _get_gpa_category(gpa):
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


def _generate_recommendations(gpa, input_data):
    """Generate personalized recommendations based on input data."""
    recommendations = []

    # Study hours recommendations
    study_hours = input_data.get('daily_study_hours', 3)
    if study_hours < 2:
        recommendations.append("Increase daily study hours to at least 2-4 hours")
    elif study_hours < 4 and gpa < 3.0:
        recommendations.append("Consider increasing study hours to 4-6 hours daily")

    # Attendance recommendations
    attendance = input_data.get('attendance_percentage', 75)
    if attendance < 60:
        recommendations.append("Critical: Improve attendance immediately - aim for above 75%")
    elif attendance < 75:
        recommendations.append("Improve attendance to above 75% for better results")

    # Stress level recommendations
    stress = input_data.get('stress_level', 'Moderate')
    if stress in ['High', 'Very_High']:
        recommendations.append("High stress detected - consider stress management techniques or counseling")

    # Motivation recommendations
    motivation = input_data.get('motivation_level', 'Motivated')
    if motivation in ['Very_Low', 'Low_Motivation']:
        recommendations.append("Consider setting clear academic goals and seeking mentorship")

    # Class participation
    participation = input_data.get('class_participation', 'Moderate')
    if participation in ['Inactive', 'Passive']:
        recommendations.append("Increase class participation for better engagement and understanding")

    # Assignment submission
    assignment = input_data.get('assignment_submission', 'Usually_On_Time')
    if assignment in ['Rarely_Submits', 'Often_Late']:
        recommendations.append("Improve assignment submission habits - submit on time consistently")

    # GPA-based recommendations
    if gpa < 2.0:
        recommendations.append("Consider academic tutoring or extra classes")
        recommendations.append("Meet with academic advisor to discuss improvement strategies")
    elif gpa < 2.5:
        recommendations.append("Focus on weak subjects and seek help from peers or tutors")

    if not recommendations:
        recommendations.append("Keep up the excellent work! Consider helping peers as a mentor.")

    return recommendations


def get_performance_summary(gpa, input_data):
    """Generate a comprehensive performance summary."""
    category = _get_gpa_category(gpa)

    # Calculate scholarship and risk data
    scholarship_data = calculate_scholarship_eligibility(gpa, input_data)
    risk_data = calculate_risk_level(gpa, input_data)
    interventions = generate_interventions(gpa, risk_data, scholarship_data, input_data)

    summary = {
        'predicted_gpa': round(gpa, 2),
        'category': category,
        'status': 'passed' if gpa >= 2.0 else 'at_risk',
        'strengths': [],
        'areas_for_improvement': [],
        'recommendations': _generate_recommendations(gpa, input_data),
        # New features
        'scholarship': scholarship_data,
        'risk_assessment': risk_data,
        'interventions': interventions
    }

    # Identify strengths
    if input_data.get('attendance_percentage', 0) >= 85:
        summary['strengths'].append('Excellent attendance')
    if input_data.get('daily_study_hours', 0) >= 4:
        summary['strengths'].append('Good study habits')
    if input_data.get('class_participation') in ['Active', 'Very_Active']:
        summary['strengths'].append('Active class participation')
    if input_data.get('motivation_level') in ['Motivated', 'Highly_Motivated']:
        summary['strengths'].append('High motivation level')
    if input_data.get('assignment_submission') in ['Always_On_Time', 'Usually_On_Time']:
        summary['strengths'].append('Consistent assignment submission')

    # Identify areas for improvement
    if input_data.get('attendance_percentage', 100) < 75:
        summary['areas_for_improvement'].append('Attendance')
    if input_data.get('daily_study_hours', 10) < 2:
        summary['areas_for_improvement'].append('Study hours')
    if input_data.get('stress_level') in ['High', 'Very_High']:
        summary['areas_for_improvement'].append('Stress management')
    if input_data.get('motivation_level') in ['Very_Low', 'Low_Motivation']:
        summary['areas_for_improvement'].append('Motivation')
    if input_data.get('assignment_submission') in ['Rarely_Submits', 'Often_Late']:
        summary['areas_for_improvement'].append('Assignment submission')

    return summary


# ============================================================================
# SCHOLARSHIP ELIGIBILITY
# ============================================================================

def calculate_scholarship_eligibility(gpa, input_data):
    """
    Calculate scholarship eligibility based on multiple criteria.

    Criteria:
    - GPA >= 3.2 AND Attendance >= 80% = Eligible
    - GPA >= 3.6 = Merit Scholarship
    - GPA >= 3.2 AND Financial need = Need-based Scholarship
    """
    attendance = input_data.get('attendance_percentage', 0)
    income = input_data.get('family_monthly_income_npr', 50000)
    assignment = input_data.get('assignment_submission', 'Usually_On_Time')

    result = {
        'eligible': False,
        'scholarship_type': None,
        'eligibility_score': 0,
        'criteria_met': [],
        'criteria_not_met': [],
        'improvement_tips': []
    }

    # Calculate eligibility score (0-100)
    score = 0

    # GPA contribution (max 50 points)
    if gpa >= 3.6:
        score += 50
        result['criteria_met'].append('Excellent GPA (3.6+)')
    elif gpa >= 3.2:
        score += 40
        result['criteria_met'].append('Good GPA (3.2+)')
    elif gpa >= 2.8:
        score += 25
        result['criteria_not_met'].append('GPA below 3.2 - Need improvement for scholarship')
    else:
        result['criteria_not_met'].append('GPA below minimum threshold (2.8)')
        result['improvement_tips'].append('Focus on improving GPA to at least 3.2')

    # Attendance contribution (max 25 points)
    if attendance >= 90:
        score += 25
        result['criteria_met'].append('Excellent attendance (90%+)')
    elif attendance >= 80:
        score += 20
        result['criteria_met'].append('Good attendance (80%+)')
    elif attendance >= 75:
        score += 10
        result['criteria_not_met'].append('Attendance below 80% - scholarship threshold')
        result['improvement_tips'].append('Improve attendance to 80%+ for scholarship eligibility')
    else:
        result['criteria_not_met'].append('Attendance below acceptable level')
        result['improvement_tips'].append('Critically low attendance - aim for 80%+')

    # Assignment submission contribution (max 15 points)
    assignment_scores = {
        'Always_On_Time': 15,
        'Usually_On_Time': 12,
        'Sometimes_Late': 8,
        'Often_Late': 4,
        'Rarely_Submits': 0
    }
    score += assignment_scores.get(assignment, 8)
    if assignment in ['Always_On_Time', 'Usually_On_Time']:
        result['criteria_met'].append('Good assignment submission record')
    else:
        result['criteria_not_met'].append('Assignment submission needs improvement')
        result['improvement_tips'].append('Submit assignments on time consistently')

    # Financial need contribution (max 10 points for need-based)
    low_income_threshold = 40000
    if income < low_income_threshold:
        score += 10
        result['criteria_met'].append('Qualifies for financial need consideration')

    result['eligibility_score'] = min(score, 100)

    # Determine eligibility and type
    if gpa >= 3.6 and attendance >= 85:
        result['eligible'] = True
        result['scholarship_type'] = 'Merit Scholarship (Full)'
    elif gpa >= 3.2 and attendance >= 80:
        result['eligible'] = True
        if income < low_income_threshold:
            result['scholarship_type'] = 'Need-Based Scholarship'
        else:
            result['scholarship_type'] = 'Merit Scholarship (Partial)'
    elif gpa >= 3.0 and attendance >= 80 and income < low_income_threshold:
        result['eligible'] = True
        result['scholarship_type'] = 'Need-Based Scholarship (Conditional)'
    else:
        result['scholarship_type'] = 'Not Eligible'
        if not result['improvement_tips']:
            result['improvement_tips'].append('Maintain GPA above 3.2 and attendance above 80%')

    return result


# ============================================================================
# RISK LEVEL CLASSIFICATION
# ============================================================================

def calculate_risk_level(gpa, input_data):
    """
    Calculate academic risk level based on multiple factors.

    Risk Levels:
    - No_Risk: GPA > 2.8 AND Attendance > 75%
    - Low_Risk: GPA 2.4-2.8 OR minor attendance issues
    - Medium_Risk: GPA 2.0-2.4 OR attendance 60-75%
    - High_Risk: GPA < 2.0 OR Attendance < 60%
    """
    attendance = input_data.get('attendance_percentage', 75)
    study_hours = input_data.get('daily_study_hours', 3)
    motivation = input_data.get('motivation_level', 'Moderately_Motivated')
    stress = input_data.get('stress_level', 'Moderate')
    assignment = input_data.get('assignment_submission', 'Usually_On_Time')
    participation = input_data.get('class_participation', 'Moderate')

    # Calculate risk score (higher = more risk)
    risk_score = 0
    risk_factors = []
    protective_factors = []

    # GPA risk assessment (0-40 points)
    if gpa < 2.0:
        risk_score += 40
        risk_factors.append({'factor': 'Critical GPA', 'severity': 'high', 'detail': f'GPA {gpa:.2f} is below passing threshold'})
    elif gpa < 2.4:
        risk_score += 30
        risk_factors.append({'factor': 'Low GPA', 'severity': 'medium', 'detail': f'GPA {gpa:.2f} is at risk of failing'})
    elif gpa < 2.8:
        risk_score += 15
        risk_factors.append({'factor': 'Borderline GPA', 'severity': 'low', 'detail': f'GPA {gpa:.2f} needs improvement'})
    else:
        protective_factors.append('Good academic standing')

    # Attendance risk (0-25 points)
    if attendance < 60:
        risk_score += 25
        risk_factors.append({'factor': 'Critical Attendance', 'severity': 'high', 'detail': f'{attendance}% attendance is critically low'})
    elif attendance < 75:
        risk_score += 15
        risk_factors.append({'factor': 'Low Attendance', 'severity': 'medium', 'detail': f'{attendance}% attendance needs improvement'})
    elif attendance >= 85:
        protective_factors.append('Excellent attendance')

    # Study hours risk (0-15 points)
    if study_hours < 1:
        risk_score += 15
        risk_factors.append({'factor': 'Insufficient Study', 'severity': 'high', 'detail': 'Less than 1 hour daily study'})
    elif study_hours < 2:
        risk_score += 10
        risk_factors.append({'factor': 'Low Study Hours', 'severity': 'medium', 'detail': 'Less than 2 hours daily study'})
    elif study_hours >= 4:
        protective_factors.append('Good study habits')

    # Motivation risk (0-10 points)
    low_motivation = ['Very_Low', 'Low_Motivation']
    if motivation in low_motivation:
        risk_score += 10
        risk_factors.append({'factor': 'Low Motivation', 'severity': 'medium', 'detail': 'Student shows low academic motivation'})
    elif motivation == 'Highly_Motivated':
        protective_factors.append('High motivation')

    # Stress risk (0-10 points)
    if stress in ['High', 'Very_High']:
        risk_score += 10
        risk_factors.append({'factor': 'High Stress', 'severity': 'medium', 'detail': 'Elevated stress levels affecting performance'})
    elif stress in ['Very_Low', 'Low']:
        protective_factors.append('Manageable stress levels')

    # Assignment submission risk (0-10 points)
    poor_submission = ['Rarely_Submits', 'Often_Late']
    if assignment in poor_submission:
        risk_score += 10
        risk_factors.append({'factor': 'Poor Submissions', 'severity': 'medium', 'detail': 'Frequent late or missing assignments'})
    elif assignment == 'Always_On_Time':
        protective_factors.append('Excellent assignment submission')

    # Class participation (0-5 points)
    if participation in ['Inactive', 'Passive']:
        risk_score += 5
        risk_factors.append({'factor': 'Low Participation', 'severity': 'low', 'detail': 'Minimal class engagement'})
    elif participation in ['Active', 'Very_Active']:
        protective_factors.append('Active class participation')

    # Determine risk level
    if risk_score >= 50:
        risk_level = 'High_Risk'
        risk_color = '#DC2626'  # Red
    elif risk_score >= 30:
        risk_level = 'Medium_Risk'
        risk_color = '#F59E0B'  # Amber
    elif risk_score >= 15:
        risk_level = 'Low_Risk'
        risk_color = '#3B82F6'  # Blue
    else:
        risk_level = 'No_Risk'
        risk_color = '#10B981'  # Green

    # Calculate dropout probability (simplified model)
    dropout_probability = min(risk_score / 100, 0.95)

    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'risk_color': risk_color,
        'dropout_probability': round(dropout_probability, 2),
        'risk_factors': risk_factors,
        'protective_factors': protective_factors,
        'total_risk_factors': len(risk_factors),
        'total_protective_factors': len(protective_factors)
    }


# ============================================================================
# INTERVENTION RECOMMENDATIONS
# ============================================================================

def generate_interventions(gpa, risk_level_data, scholarship_data, input_data):
    """
    Generate personalized intervention recommendations based on risk level and performance.

    Intervention Categories:
    - Immediate: Critical actions needed now
    - Short-term: Actions for the current semester
    - Long-term: Career and academic planning
    """
    risk_level = risk_level_data['risk_level']
    risk_factors = risk_level_data['risk_factors']

    interventions = {
        'immediate': [],
        'short_term': [],
        'long_term': [],
        'support_services': [],
        'priority_level': 'normal'
    }

    # High Risk Interventions
    if risk_level == 'High_Risk':
        interventions['priority_level'] = 'critical'
        interventions['immediate'].extend([
            {'action': 'Schedule meeting with academic advisor', 'urgency': 'critical'},
            {'action': 'Enroll in mandatory tutoring program', 'urgency': 'critical'},
            {'action': 'Create daily study schedule with accountability partner', 'urgency': 'high'}
        ])
        interventions['support_services'].extend([
            'Academic Counseling Center',
            'Peer Tutoring Program',
            'Student Success Coach'
        ])

    # Medium Risk Interventions
    elif risk_level == 'Medium_Risk':
        interventions['priority_level'] = 'elevated'
        interventions['immediate'].extend([
            {'action': 'Meet with course instructor during office hours', 'urgency': 'high'},
            {'action': 'Join study group for challenging subjects', 'urgency': 'medium'}
        ])
        interventions['support_services'].extend([
            'Study Skills Workshop',
            'Time Management Seminar'
        ])

    # Low Risk Interventions
    elif risk_level == 'Low_Risk':
        interventions['priority_level'] = 'monitor'
        interventions['short_term'].extend([
            {'action': 'Review weak subject areas', 'urgency': 'medium'},
            {'action': 'Increase daily study hours by 30 minutes', 'urgency': 'low'}
        ])

    # No Risk - Enhancement Interventions
    else:
        interventions['priority_level'] = 'enhancement'
        interventions['long_term'].extend([
            {'action': 'Consider leadership roles in student organizations', 'urgency': 'low'},
            {'action': 'Explore research opportunities with faculty', 'urgency': 'low'},
            {'action': 'Apply for competitive internships', 'urgency': 'medium'}
        ])

    # Factor-specific interventions
    for factor in risk_factors:
        if factor['factor'] == 'Critical Attendance' or factor['factor'] == 'Low Attendance':
            interventions['immediate'].append({
                'action': 'Set up attendance tracking system with daily reminders',
                'urgency': 'high'
            })
            interventions['support_services'].append('Attendance Monitoring Program')

        if factor['factor'] == 'Insufficient Study' or factor['factor'] == 'Low Study Hours':
            interventions['short_term'].append({
                'action': 'Use Pomodoro technique - start with 25-minute focused sessions',
                'urgency': 'medium'
            })
            interventions['short_term'].append({
                'action': 'Find quiet study space in library or study room',
                'urgency': 'medium'
            })

        if factor['factor'] == 'Low Motivation':
            interventions['short_term'].append({
                'action': 'Meet with career counselor to clarify goals',
                'urgency': 'medium'
            })
            interventions['support_services'].append('Career Counseling Services')

        if factor['factor'] == 'High Stress':
            interventions['immediate'].append({
                'action': 'Practice stress management techniques (meditation, exercise)',
                'urgency': 'high'
            })
            interventions['support_services'].append('Mental Health Counseling')
            interventions['support_services'].append('Stress Management Workshop')

        if factor['factor'] == 'Poor Submissions':
            interventions['immediate'].append({
                'action': 'Create assignment calendar with deadlines',
                'urgency': 'high'
            })
            interventions['short_term'].append({
                'action': 'Start assignments early - aim for 50% completion 3 days before deadline',
                'urgency': 'medium'
            })

    # Scholarship-related interventions
    if not scholarship_data['eligible'] and gpa >= 2.8:
        interventions['long_term'].append({
            'action': 'Work towards scholarship eligibility - maintain GPA above 3.2',
            'urgency': 'medium'
        })
        for tip in scholarship_data['improvement_tips']:
            interventions['short_term'].append({
                'action': tip,
                'urgency': 'medium'
            })

    # GPA-specific recommendations
    if gpa < 2.0:
        interventions['immediate'].append({
            'action': 'Consider reduced course load next semester',
            'urgency': 'high'
        })
        interventions['immediate'].append({
            'action': 'Identify and focus on easiest courses to pass first',
            'urgency': 'critical'
        })
    elif gpa < 2.5:
        interventions['short_term'].append({
            'action': 'Attend all review sessions and office hours',
            'urgency': 'high'
        })
    elif gpa >= 3.2:
        interventions['long_term'].append({
            'action': 'Consider becoming a peer tutor to reinforce knowledge',
            'urgency': 'low'
        })
        interventions['long_term'].append({
            'action': 'Explore honors program or advanced courses',
            'urgency': 'low'
        })

    # Remove duplicates from support services
    interventions['support_services'] = list(set(interventions['support_services']))

    return interventions
