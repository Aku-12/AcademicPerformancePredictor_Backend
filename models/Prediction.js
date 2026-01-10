const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
  student: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Student',
    required: true
  },
  inputData: {
    studyHoursPerWeek: Number,
    attendancePercentage: Number,
    previousGpa: Number,
    assignmentsCompleted: Number,
    classParticipation: Number,
    extracurricularActivities: Number,
    sleepHours: Number,
    stressLevel: String
  },
  predictedGpa: {
    type: Number,
    required: true,
    min: 0,
    max: 4.0
  },
  gpaCategory: {
    type: String,
    enum: ['Excellent', 'Very Good', 'Good', 'Satisfactory', 'Needs Improvement', 'At Risk']
  },
  recommendations: [{
    type: String
  }],
  modelType: {
    type: String,
    default: 'random_forest'
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

// Index for faster queries
predictionSchema.index({ student: 1, createdAt: -1 });

// Static method to get prediction history for a student
predictionSchema.statics.getStudentHistory = function(studentId, limit = 10) {
  return this.find({ student: studentId })
    .sort({ createdAt: -1 })
    .limit(limit);
};

// Static method to get average predicted GPA
predictionSchema.statics.getAveragePredictedGpa = async function(studentId) {
  const result = await this.aggregate([
    { $match: { student: new mongoose.Types.ObjectId(studentId) } },
    { $group: { _id: null, avgGpa: { $avg: '$predictedGpa' } } }
  ]);
  return result[0]?.avgGpa || null;
};

module.exports = mongoose.model('Prediction', predictionSchema);
