import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import PatientDashboard from './pages/patients/Dashboard';
import DataInput from './pages/patients/DataInput';
import Recommendations from './pages/patients/Recommendations';
import Login from './components/auth/Login';
import ProtectedRoute from './components/auth/ProtectedRoute';
import Alerts from './pages/patients/Alerts';
import Metrics from './pages/patients/Metrics';
import Feedback from './pages/patients/Feedback';
import Navbar from './components/navbar/navbar';

function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/login" element={<Login />} />

        {/* Protected routes for patient module */}
        <Route
          path="patient/dashboard"
          element={
            <ProtectedRoute>
              <PatientDashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="patient/data-input"
          element={
            <ProtectedRoute>
              <DataInput />
            </ProtectedRoute>
          }
        />
        <Route
          path="patient/recommendations"
          element={
            <ProtectedRoute>
              <Recommendations />
            </ProtectedRoute>
          }
        />
        <Route
          path="patient/alerts"
          element={
            <ProtectedRoute>
              <Alerts />
            </ProtectedRoute>
          }
        />
        <Route
          path="patient/metrics"
          element={
            <ProtectedRoute>
              <Metrics />
            </ProtectedRoute>
          }
        />
        <Route
          path="patient/feedback"
          element={
            <ProtectedRoute>
              <Feedback />
            </ProtectedRoute>
          }
        />

        {/* Redirect unknown routes to login */}
        <Route path="*" element={<Navigate to="/login" />} />
      </Routes>
    </>
  );
}

export default App;