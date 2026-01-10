const Student = require('../models/Student');
const Prediction = require('../models/Prediction');
const mlService = require('../services/mlService');

// Predict GPA for a student
exports.predictGpa = async (req, res) => {
  try {
    const { studentId } = req.params;

    // Find student
    const student = await Student.findById(studentId);
    if (!student) {
      return res.status(404).json({
        success: false,
        error: 'Student not found'
      });
    }

    // Convert student data to ML format
    const mlData = student.toMLFormat();

    // Get prediction from ML service
    const mlResponse = await mlService.predict(mlData);

    // Save prediction to database
    const prediction = new Prediction({
      student: student._id,
      inputData: {
        studyHoursPerWeek: student.academicData.studyHoursPerWeek,
        attendancePercentage: student.academicData.attendancePercentage,
        previousGpa: student.academicData.previousGpa,
        assignmentsCompleted: student.academicData.assignmentsCompleted,
        classParticipation: student.academicData.classParticipation,
        extracurricularActivities: student.academicData.extracurricularActivities,
        sleepHours: student.academicData.sleepHours,
        stressLevel: student.academicData.stressLevel
      },
      predictedGpa: mlResponse.predicted_gpa,
      gpaCategory: mlResponse.gpa_category,
      recommendations: mlResponse.recommendations || []
    });

    await prediction.save();

    res.json({
      success: true,
      data: {
        student: {
          id: student._id,
          name: `${student.firstName} ${student.lastName}`,
          studentId: student.studentId
        },
        prediction: {
          predictedGpa: prediction.predictedGpa,
          gpaCategory: prediction.gpaCategory,
          recommendations: prediction.recommendations,
          createdAt: prediction.createdAt
        }
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// Predict with custom data (without saving student)
exports.predictCustom = async (req, res) => {
  try {
    const mlData = {
      study_hours_per_week: req.body.studyHoursPerWeek,
      attendance_percentage: req.body.attendancePercentage,
      previous_gpa: req.body.previousGpa,
      assignments_completed: req.body.assignmentsCompleted,
      class_participation: req.body.classParticipation,
      extracurricular_activities: req.body.extracurricularActivities || 0,
      sleep_hours: req.body.sleepHours || 7,
      stress_level: req.body.stressLevel || 'medium'
    };

    const mlResponse = await mlService.predict(mlData);

    res.json({
      success: true,
      data: {
        predictedGpa: mlResponse.predicted_gpa,
        gpaCategory: mlResponse.gpa_category,
        recommendations: mlResponse.recommendations || []
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// Get prediction history for a student
exports.getPredictionHistory = async (req, res) => {
  try {
    const { studentId } = req.params;
    const { limit = 10 } = req.query;

    const student = await Student.findById(studentId);
    if (!student) {
      return res.status(404).json({
        success: false,
        error: 'Student not found'
      });
    }

    const predictions = await Prediction.getStudentHistory(studentId, parseInt(limit));

    res.json({
      success: true,
      data: {
        student: {
          id: student._id,
          name: `${student.firstName} ${student.lastName}`
        },
        predictions
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// Get prediction by ID
exports.getPredictionById = async (req, res) => {
  try {
    const prediction = await Prediction.findById(req.params.id).populate('student', 'firstName lastName studentId email');

    if (!prediction) {
      return res.status(404).json({
        success: false,
        error: 'Prediction not found'
      });
    }

    res.json({
      success: true,
      data: prediction
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// Get all predictions with pagination
exports.getAllPredictions = async (req, res) => {
  try {
    const { page = 1, limit = 10 } = req.query;

    const predictions = await Prediction.find()
      .populate('student', 'firstName lastName studentId')
      .skip((page - 1) * limit)
      .limit(parseInt(limit))
      .sort({ createdAt: -1 });

    const total = await Prediction.countDocuments();

    res.json({
      success: true,
      data: predictions,
      pagination: {
        current: parseInt(page),
        pages: Math.ceil(total / limit),
        total
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// Get student statistics
exports.getStudentStats = async (req, res) => {
  try {
    const { studentId } = req.params;

    const student = await Student.findById(studentId);
    if (!student) {
      return res.status(404).json({
        success: false,
        error: 'Student not found'
      });
    }

    const avgGpa = await Prediction.getAveragePredictedGpa(studentId);
    const predictions = await Prediction.getStudentHistory(studentId, 5);
    const totalPredictions = await Prediction.countDocuments({ student: studentId });

    res.json({
      success: true,
      data: {
        student: {
          id: student._id,
          name: `${student.firstName} ${student.lastName}`,
          studentId: student.studentId
        },
        stats: {
          averagePredictedGpa: avgGpa ? avgGpa.toFixed(2) : null,
          totalPredictions,
          recentPredictions: predictions
        }
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// Train model
exports.trainModel = async (req, res) => {
  try {
    const { modelType = 'random_forest', nSamples = 1000 } = req.body;

    const result = await mlService.trainModel({
      model_type: modelType,
      n_samples: nSamples
    });

    res.json({
      success: true,
      message: 'Model trained successfully',
      data: result
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// Get ML service health
exports.getMLHealth = async (req, res) => {
  try {
    const health = await mlService.healthCheck();
    res.json({
      success: true,
      data: health
    });
  } catch (error) {
    res.status(503).json({
      success: false,
      error: 'ML service unavailable',
      details: error.message
    });
  }
};

// Get model info
exports.getModelInfo = async (req, res) => {
  try {
    const info = await mlService.getModelInfo();
    res.json({
      success: true,
      data: info
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};
