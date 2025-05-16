import React, { useState } from 'react';
import { 
  Typography, 
  Slider, 
  Grid, 
  Card, 
  CardContent,
  Box,
  Paper,
  Divider,
  LinearProgress,
  Chip
} from '@mui/material';
import { 
  Restaurant,
  FitnessCenter,
  ShowChart
} from '@mui/icons-material';

function Feedback() {
  const [dietaryPreference, setDietaryPreference] = useState(5);
  const [exerciseLevel, setExerciseLevel] = useState(30);
  const [impact, setImpact] = useState({
    bmi: 24.5,
    glucose: 95,
    sleep: 7.2
  });

  const handleDietaryChange = (event, newValue) => {
    setDietaryPreference(newValue);
    // Simulate impact calculation
    setImpact({
      bmi: (24.5 + (newValue - 5) * 0.2).toFixed(1),
      glucose: (95 + (newValue - 5) * 3).toFixed(0),
      sleep: (7.2 - (newValue - 5) * 0.1).toFixed(1)
    });
  };

  const handleExerciseChange = (event, newValue) => {
    setExerciseLevel(newValue);
    // Simulate impact calculation
    setImpact({
      bmi: (impact.bmi - (newValue - 30) * 0.01).toFixed(1),
      glucose: (impact.glucose - (newValue - 30) * 0.2).toFixed(0),
      sleep: (parseFloat(impact.sleep) + (newValue - 30) * 0.02).toFixed(1)
    });
  };

  const getDietQuality = (value) => {
    if (value <= 3) return 'Excellent';
    if (value <= 6) return 'Good';
    if (value <= 8) return 'Fair';
    return 'Poor';
  };

  const getExerciseLevel = (value) => {
    if (value < 15) return 'Sedentary';
    if (value < 30) return 'Light';
    if (value < 60) return 'Moderate';
    return 'Active';
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ 
        fontWeight: 600,
        mb: 3,
        color: 'primary.main'
      }}>
        Health Feedback Simulator
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card sx={{ 
            borderRadius: 3,
            height: '100%',
            boxShadow: 3
          }}>
            <CardContent sx={{ p: 3 }}>
              <Box display="flex" alignItems="center" mb={2}>
                <Restaurant color="primary" sx={{ fontSize: 32, mr: 1.5 }} />
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Dietary Preferences
                </Typography>
              </Box>
              
              <Box mb={3}>
                <Typography gutterBottom>
                  Diet Quality: <Chip 
                    label={getDietQuality(dietaryPreference)} 
                    color={
                      dietaryPreference <= 3 ? 'success' : 
                      dietaryPreference <= 6 ? 'primary' : 
                      dietaryPreference <= 8 ? 'warning' : 'error'
                    } 
                    size="small" 
                  />
                </Typography>
                <Slider
                  value={dietaryPreference}
                  onChange={handleDietaryChange}
                  valueLabelDisplay="auto"
                  step={1}
                  marks={[
                    { value: 1, label: 'Best' },
                    { value: 10, label: 'Worst' },
                  ]}
                  min={1}
                  max={10}
                  sx={{ mt: 3 }}
                />
              </Box>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" gutterBottom>
                Projected Impact:
              </Typography>
              <Box>
                <Typography variant="body2">
                  BMI: {impact.bmi} ({dietaryPreference <= 5 ? '↓' : '↑'} from 24.5)
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={Math.min(100, dietaryPreference * 10)} 
                  color={
                    dietaryPreference <= 3 ? 'success' : 
                    dietaryPreference <= 6 ? 'primary' : 
                    dietaryPreference <= 8 ? 'warning' : 'error'
                  }
                  sx={{ height: 8, borderRadius: 4, mt: 1, mb: 2 }}
                />
                
                <Typography variant="body2">
                  Glucose: {impact.glucose} mg/dL ({dietaryPreference <= 5 ? '↓' : '↑'} from 95)
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={Math.min(100, dietaryPreference * 10)} 
                  color={
                    dietaryPreference <= 3 ? 'success' : 
                    dietaryPreference <= 6 ? 'primary' : 
                    dietaryPreference <= 8 ? 'warning' : 'error'
                  }
                  sx={{ height: 8, borderRadius: 4, mt: 1 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ 
            borderRadius: 3,
            height: '100%',
            boxShadow: 3
          }}>
            <CardContent sx={{ p: 3 }}>
              <Box display="flex" alignItems="center" mb={2}>
                <FitnessCenter color="primary" sx={{ fontSize: 32, mr: 1.5 }} />
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Exercise Level
                </Typography>
              </Box>
              
              <Box mb={3}>
                <Typography gutterBottom>
                  Activity Level: <Chip 
                    label={getExerciseLevel(exerciseLevel)} 
                    color={
                      exerciseLevel < 15 ? 'error' : 
                      exerciseLevel < 30 ? 'warning' : 
                      exerciseLevel < 60 ? 'primary' : 'success'
                    } 
                    size="small" 
                  />
                </Typography>
                <Slider
                  value={exerciseLevel}
                  onChange={handleExerciseChange}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${value} min`}
                  step={5}
                  marks={[
                    { value: 0, label: '0' },
                    { value: 60, label: '60' },
                    { value: 120, label: '120' },
                  ]}
                  min={0}
                  max={120}
                  sx={{ mt: 3 }}
                />
              </Box>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" gutterBottom>
                Projected Impact:
              </Typography>
              <Box>
                <Typography variant="body2">
                  Sleep Quality: {impact.sleep} hrs ({exerciseLevel >= 30 ? '↑' : '↓'} from 7.0)
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={Math.min(100, exerciseLevel / 1.2)} 
                  color={
                    exerciseLevel < 15 ? 'error' : 
                    exerciseLevel < 30 ? 'warning' : 
                    exerciseLevel < 60 ? 'primary' : 'success'
                  }
                  sx={{ height: 8, borderRadius: 4, mt: 1, mb: 2 }}
                />
                
                <Typography variant="body2">
                  Energy Level: {Math.min(10, Math.floor(exerciseLevel / 12))}/10
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={Math.min(100, exerciseLevel / 1.2)} 
                  color={
                    exerciseLevel < 15 ? 'error' : 
                    exerciseLevel < 30 ? 'warning' : 
                    exerciseLevel < 60 ? 'primary' : 'success'
                  }
                  sx={{ height: 8, borderRadius: 4, mt: 1 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Feedback;