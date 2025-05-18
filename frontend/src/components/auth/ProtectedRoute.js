// src/components/auth/ProtectedRoute.js
import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../../authContext';

const ProtectedRoute = ({ children }) => {
  const { authState } = useAuth();

  if (!authState.isAuthenticated) {
    // Redirect to login page if not authenticated
    return <Navigate to="/login" />;
  }

  // Render the child component if authenticated
  return children;
};

export default ProtectedRoute;