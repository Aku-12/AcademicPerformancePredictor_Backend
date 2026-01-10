const express = require('express');
const router = express.Router();
const predictionController = require('../controllers/predictionController');

// ML Service routes
router.get('/health', predictionController.getMLHealth);
router.get('/model/info', predictionController.getModelInfo);
router.post('/train', predictionController.trainModel);

// Prediction routes
router.post('/predict/custom', predictionController.predictCustom);
router.post('/predict/:studentId', predictionController.predictGpa);

// Get all predictions
router.get('/', predictionController.getAllPredictions);

// Get prediction by ID
router.get('/:id', predictionController.getPredictionById);

// Student-specific routes
router.get('/student/:studentId/history', predictionController.getPredictionHistory);
router.get('/student/:studentId/stats', predictionController.getStudentStats);

module.exports = router;
