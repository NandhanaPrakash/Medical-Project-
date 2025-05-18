import React, { useState, useEffect } from 'react';
import { 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  CircularProgress, 
  Alert,
  Box,
  Paper,
  Divider,
  Tabs,
  Tab,
  useTheme
} from '@mui/material';
import { 
  MonitorHeart, 
  Bloodtype, 
  Bedtime, 
  TrendingUp,
  FitnessCenter,
  LocalDining,
  WaterDrop,
  Scale,
  HeartBroken
} from '@mui/icons-material';
import { getPatientData, getHealthMetrics } from '../../services/api';
import { useAuth } from '../../authContext';
import HealthStatusCard from '../../components/dashboard/HealthStatusCard';

function Dashboard({ data }) {
  if (!data) {
    return (
      <Paper elevation={3} sx={{ p: 3, mt: 4 }}>
        <Typography variant="h6">No data submitted yet.</Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 3, mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        Health Dashboard
      </Typography>
      {Object.entries(data).map(([key, value]) => (
        value && (
          <Typography key={key} sx={{ mb: 1 }}>
            <strong>
              {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}:
            </strong>{' '}
            {value}
          </Typography>
        )
      ))}
    </Paper>
  );
}

function PatientDashboard() {
  const [patientInfo, setPatientInfo] = useState(null);
  const [healthData, setHealthData] = useState({
    bmi: [],
    glucose: [],
    sleep: [],
    activity: [],
    calories: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const { authState } = useAuth();
  const patientId = authState?.patientId;
  const theme = useTheme();

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        if (patientId) {
          const [patientData, bmiData, glucoseData, sleepData, activityData, calorieData] = await Promise.all([
            getPatientData(patientId),
            getHealthMetrics(patientId, 'bmi', 'monthly'),
            getHealthMetrics(patientId, 'glucose', 'weekly'),
            getHealthMetrics(patientId, 'sleep', 'daily'),
            getHealthMetrics(patientId, 'activity', 'weekly'),
            getHealthMetrics(patientId, 'calories', 'daily')
          ]);
          
          setPatientInfo(patientData);
          setHealthData({
            bmi: bmiData,
            glucose: glucoseData,
            sleep: sleepData,
            activity: activityData,
            calories: calorieData
          });
        }
      } catch (err) {
        console.error('Failed to fetch data', err);
        setError('Failed to load health data. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [patientId]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!patientInfo) {
    return (
      <Typography variant="body1" color="textSecondary">
        No patient data available.
      </Typography>
    );
  }

  const healthMetrics = [
    { 
      title: 'BMI', 
      value: patientInfo.bmi, 
      icon: <Scale color="primary" sx={{ fontSize: 40 }} />,
      status: patientInfo.bmi < 25 ? 'Normal' : 'High',
      unit: '',
      trend: healthData.bmi,
      idealRange: [18.5, 24.9]
    },
    { 
      title: 'Glucose', 
      value: patientInfo.glucoseLevel, 
      icon: <Bloodtype color="primary" sx={{ fontSize: 40 }} />,
      status: patientInfo.glucoseLevel < 100 ? 'Normal' : 'Monitor',
      unit: 'mg/dL',
      trend: healthData.glucose,
      idealRange: [70, 99]
    },
    { 
      title: 'Sleep', 
      value: patientInfo.sleepHours, 
      icon: <Bedtime color="primary" sx={{ fontSize: 40 }} />,
      status: patientInfo.sleepHours >= 7 ? 'Good' : 'Needs Improvement',
      unit: 'hrs',
      trend: healthData.sleep,
      idealRange: [7, 9]
    },
    { 
      title: 'Activity', 
      value: patientInfo.physicalActivity, 
      icon: <FitnessCenter color="primary" sx={{ fontSize: 40 }} />,
      status: 'Active',
      unit: '',
      trend: healthData.activity,
      idealRange: null
    },
    { 
      title: 'Calories', 
      value: patientInfo.calorieIntake, 
      icon: <LocalDining color="primary" sx={{ fontSize: 40 }} />,
      status: patientInfo.calorieIntake > 2000 ? 'High' : 'Balanced',
      unit: 'kcal',
      trend: healthData.calories,
      idealRange: [1800, 2200]
    },
    { 
      title: 'Hydration', 
      value: patientInfo.waterIntake || '--', 
      icon: <WaterDrop color="primary" sx={{ fontSize: 40 }} />,
      status: patientInfo.waterIntake >= 2000 ? 'Good' : 'Needs More',
      unit: 'ml',
      trend: [],
      idealRange: [2000, 3000]
    }
  ];

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ 
        fontWeight: 600,
        mb: 4,
        color: 'primary.main'
      }}>
        Health Dashboard
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {healthMetrics.map((metric, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <HealthStatusCard 
              title={metric.title}
              value={metric.value}
              unit={metric.unit}
              icon={metric.icon}
              status={metric.status}
              trendData={metric.trend}
              idealRange={metric.idealRange}
            />
          </Grid>
        ))}
      </Grid>

      <Paper elevation={3} sx={{ 
        p: 3, 
        borderRadius: 3,
        mb: 4
      }}>
        <Box display="flex" alignItems="center" mb={3}>
          <TrendingUp color="primary" sx={{ fontSize: 32, mr: 1.5 }} />
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            Health Status Overview
          </Typography>
        </Box>
        <Divider sx={{ mb: 3 }} />

        <Tabs 
          value={activeTab} 
          onChange={handleTabChange} 
          variant="scrollable"
          scrollButtons="auto"
          sx={{ mb: 3 }}
        >
          <Tab label="BMI" icon={<Scale />} />
          <Tab label="Glucose" icon={<Bloodtype />} />
          <Tab label="Sleep" icon={<Bedtime />} />
          <Tab label="Activity" icon={<FitnessCenter />} />
          <Tab label="Calories" icon={<LocalDining />} />
        </Tabs>

        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
          {healthMetrics[activeTab].title} Overview
        </Typography>
        <Typography variant="body1" sx={{ color: 'text.secondary' }}>
          {healthMetrics[activeTab].value} {healthMetrics[activeTab].unit} 
          {healthMetrics[activeTab].status && (
            <span> - Status: {healthMetrics[activeTab].status}</span>
          )}
        </Typography>
      </Paper>

      {/* Add the Dashboard component for form data display */}
      {patientInfo.formData && (
        <Paper elevation={3} sx={{ p: 3, mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Submitted Health Data
          </Typography>
          <Dashboard data={patientInfo.formData} />
        </Paper>
      )}
    </Box>
  );
}

export default PatientDashboard;