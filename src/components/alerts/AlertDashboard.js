import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Alert } from '@mui/material';

function AlertDashboard() {
  // Replace with real-time alert fetching or state updates
  const [alerts, setAlerts] = useState([
    { id: 1, type: 'High Glucose', message: 'Your glucose level is significantly higher than normal.', severity: 'error', timestamp: new Date().toLocaleTimeString() },
    { id: 2, type: 'Abnormal BMI', message: 'Your BMI has moved into an unhealthy range.', severity: 'warning', timestamp: new Date(Date.now() - 3600000).toLocaleTimeString() },
  ]);

  return (
    <div>
      <Typography variant="h6" gutterBottom>
        Real-Time Health Alerts
      </Typography>
      {alerts.length > 0 ? (
        alerts.map((alert) => (
          <Alert key={alert.id} severity={alert.severity} sx={{ mb: 1 }}>
            {alert.type}: {alert.message} ({alert.timestamp})
          </Alert>
        ))
      ) : (
        <Typography variant="body2">No new alerts at this time.</Typography>
      )}
    </div>
  );
}

export default AlertDashboard;