const mongoose = require('mongoose');

const studentSchema = new mongoose.Schema({
  studentId: {
    type: String,
    required: true,
    unique: true
  },
  firstName: {
    type: String,
    required: true,
    trim: true
  },
  lastName: {
    type: String,
    required: true,
    trim: true
  },
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true
  },
  department: {
    type: String,
    required: true
  },
  semester: {
    type: Number,
    required: true,
    min: 1,
    max: 8
  },
  academicData: {
    studyHoursPerWeek: {
      type: Number,
      min: 0,
      max: 168,
      default: 0
    },
    attendancePercentage: {
      type: Number,
      min: 0,
      max: 100,
      default: 0
    },
    previousGpa: {
      type: Number,
      min: 0,
      max: 4.0,
      default: 0
    },
    assignmentsCompleted: {
      type: Number,
      min: 0,
      max: 100,
      default: 0
    },
    classParticipation: {
      type: Number,
      min: 0,
      max: 10,
      default: 0
    },
    extracurricularActivities: {
      type: Number,
      min: 0,
      max: 10,
      default: 0
    },
    sleepHours: {
      type: Number,
      min: 0,
      max: 24,
      default: 7
    },
    stressLevel: {
      type: String,
      enum: ['low', 'medium', 'high'],
      default: 'medium'
    }
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// Update timestamp on save
studentSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Convert to ML format
studentSchema.methods.toMLFormat = function() {
  return {
    study_hours_per_week: this.academicData.studyHoursPerWeek,
    attendance_percentage: this.academicData.attendancePercentage,
    previous_gpa: this.academicData.previousGpa,
    assignments_completed: this.academicData.assignmentsCompleted,
    class_participation: this.academicData.classParticipation,
    extracurricular_activities: this.academicData.extracurricularActivities,
    sleep_hours: this.academicData.sleepHours,
    stress_level: this.academicData.stressLevel
  };
};

module.exports = mongoose.model('Student', studentSchema);
