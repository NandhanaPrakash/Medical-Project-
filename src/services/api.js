// src/services/api.js
import axios from 'axios';
import patientsData from '../data/patients.json';

const API_BASE_URL = '/api'; // Placeholder - not really used for patient data now

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// --- Utility Function for Setting Auth Token (Simulated) ---
export const setAuthToken = (token) => {
  // In a real scenario, you'd handle token storage here
  console.log('Simulated setAuthToken:', token);
};

// --- Authentication (Simulated with local JSON) ---
export const login = async (credentials) => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      const patient = Object.values(patientsData).find(
        (p) => p.username === credentials.username && p.password === credentials.password
      );
      if (patient) {
        // Simulate a token
        const token = `fake-token-${patient.id}`;
        resolve({ token, patientId: patient.id });
      } else {
        reject({ message: 'Invalid username or password' });
      }
    }, 500); // Simulate API delay
  });
};

// --- Patient Data Management (Simulated with local JSON) ---
export const getPatientData = async (patientId) => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (patientsData[patientId]) {
        resolve(patientsData[patientId]);
      } else {
        reject({ message: `Patient data not found for ID: ${patientId}` });
      }
    }, 300); // Simulate API delay
  });
};

export const updatePatientData = async (patientId, data) => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (patientsData[patientId]) {
        patientsData[patientId] = { ...patientsData[patientId], ...data };
        // In a real scenario, you'd update the backend here
        console.log('Simulated patient data update:', patientsData[patientId]);
        resolve(patientsData[patientId]);
      } else {
        reject({ message: `Patient not found for ID: ${patientId}` });
      }
    }, 500); // Simulate API delay
  });
};

export const logHealthData = async (patientId, data) => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (patientsData[patientId]) {
        patientsData[patientId] = {
          ...patientsData[patientId],
          ...data,
          lastUpdated: new Date().toISOString(),
        };
        console.log('Simulated health data log:', patientsData[patientId]);
        resolve(patientsData[patientId]);
      } else {
        reject({ message: `Patient not found for ID: ${patientId}` });
      }
    }, 400); // Simulate API delay
  });
};

// --- Health Recommendation System (Simulated - will be updated later) ---
export const getRecommendations = async (patientId) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Simulate recommendations based on patient ID (you'll expand this later)
      const recommendations = patientsData[patientId]?.bmi > 25
        ? [{ id: 1, type: 'Diet', text: 'Consider a weight management diet.' }]
        : [{ id: 2, type: 'Activity', text: 'Maintain regular physical activity.' }];
      resolve(recommendations);
    }, 400);
  });
};

export const acceptRecommendation = async (recommendationId) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      console.log('Simulated recommendation accepted:', recommendationId);
      resolve({ message: `Recommendation ${recommendationId} accepted.` });
    }, 300);
  });
};

export const requestNewRecommendation = async (patientId, preferences) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      console.log('Simulated request for new recommendation:', patientId, preferences);
      resolve([{ id: 3, type: 'Lifestyle', text: 'Consider stress reduction techniques.' }]);
    }, 500);
  });
};

// --- Health Anomaly Detection (Simulated - will be updated later) ---
export const getAlerts = async (patientId) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const alerts = patientsData[patientId]?.glucoseLevel > 150
        ? [{ id: 1, type: 'High Glucose', message: 'Glucose level is elevated.', severity: 'warning' }]
        : [];
      resolve(alerts);
    }, 350);
  });
};

export const getAnomalyLog = async (patientId) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const logs = patientsData[patientId]?.glucoseLevel > 150
        ? [{ id: 1, metric: 'Glucose', value: patientsData[patientId].glucoseLevel, timestamp: new Date().toISOString(), severity: 'High' }]
        : [];
      resolve(logs);
    }, 450);
  });
};

// --- Health Metrics Visualization (Simulated - will be updated later) ---
export const getHealthMetrics = async (patientId, metric, timePeriod) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const dummyData = [];
      const endDate = new Date();
      let startDate = new Date();
      if (timePeriod === 'weekly') startDate.setDate(endDate.getDate() - 7);
      if (timePeriod === 'monthly') startDate.setMonth(endDate.getMonth() - 1);

      let currentDate = new Date(startDate);
      while (currentDate <= endDate) {
        const value = patientsData[patientId]?.[metric] || Math.random() * 100;
        dummyData.push({ date: currentDate.toISOString().split('T')[0], value: parseFloat(value.toFixed(1)) });
        currentDate.setDate(currentDate.getDate() + 1);
      }
      resolve(dummyData);
    }, 500);
  });
};

// --- Real-Time Health Feedback System (Simulated - will be updated later) ---
export const simulateBehaviorChange = async (patientId, behaviorData) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      console.log('Simulated behavior change:', patientId, behaviorData);
      resolve({ message: 'Behavior change simulated.' });
    }, 400);
  });
};

export default api;