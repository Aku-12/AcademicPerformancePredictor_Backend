const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5001';

class MLService {
  async predict(studentData) {
    const response = await fetch(`${ML_SERVICE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(studentData)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Prediction failed');
    }

    return response.json();
  }

  async predictBatch(studentsData) {
    const response = await fetch(`${ML_SERVICE_URL}/predict/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(studentsData)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Batch prediction failed');
    }

    return response.json();
  }

  async trainModel(options = {}) {
    const response = await fetch(`${ML_SERVICE_URL}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Training failed');
    }

    return response.json();
  }

  async getModelInfo() {
    const response = await fetch(`${ML_SERVICE_URL}/model/info`);

    if (!response.ok) {
      throw new Error('Failed to get model info');
    }

    return response.json();
  }

  async healthCheck() {
    const response = await fetch(`${ML_SERVICE_URL}/health`);

    if (!response.ok) {
      throw new Error('ML service unhealthy');
    }

    return response.json();
  }
}

module.exports = new MLService();
