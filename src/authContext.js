// src/authContext.js
import React, { createContext, useState, useContext, useEffect } from 'react';
import { login as loginService } from './services/api';
import { useNavigate } from 'react-router-dom';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [authState, setAuthState] = useState({
    isAuthenticated: false,
    patientId: localStorage.getItem('patientId') || null,
    token: localStorage.getItem('authToken') || null,
  });
  const navigate = useNavigate();

  useEffect(() => {
    // Check for stored token on initial load
    if (authState.token && authState.patientId) {
      setAuthState(prev => ({ ...prev, isAuthenticated: true }));
    }
  }, []);

  const login = async (credentials) => {
    try {
      const data = await loginService(credentials);
      setAuthState({ isAuthenticated: true, patientId: data.patientId, token: data.token });
      localStorage.setItem('authToken', data.token);
      localStorage.setItem('patientId', data.patientId);
      navigate('/patient/dashboard');
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  };

  const logout = () => {
    setAuthState({ isAuthenticated: false, patientId: null, token: null });
    localStorage.removeItem('authToken');
    localStorage.removeItem('patientId');
    navigate('/login');
  };

  const value = {
    authState,
    login,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);