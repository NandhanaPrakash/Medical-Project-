import React, { useState } from 'react';
import {
  TextField,
  Button,
  Grid,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Paper,
  Divider,
  Stepper,
  Step,
  StepLabel,
  MobileStepper
} from '@mui/material';
import {
  FitnessCenter,
  LocalDining,
  Bedtime,
  MonitorHeart,
  CheckCircle
} from '@mui/icons-material';

function DataInputForm({ onSubmit }) {
  const [activeStep, setActiveStep] = useState(0);
  const [formData, setFormData] = useState({
    activityType: '',
    activityDuration: '',
    calories: '',
    waterIntake: '',
    sleepHours: '',
    sleepQuality: '',
    glucoseLevel: '',
    bloodPressure: '',
    notes: ''
  });

  const steps = ['Activity', 'Nutrition', 'Sleep', 'Health Metrics', 'Review'];

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleChange = (field) => (event) => {
    setFormData({ ...formData, [field]: event.target.value });
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (onSubmit) {
      onSubmit(formData);
    }
    alert('Form Submitted Successfully!');
  };

  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Activity Type</InputLabel>
                <Select
                  value={formData.activityType}
                  onChange={handleChange('activityType')}
                  label="Activity Type"
                >
                  <MenuItem value="walking">Walking</MenuItem>
                  <MenuItem value="running">Running</MenuItem>
                  <MenuItem value="cycling">Cycling</MenuItem>
                  <MenuItem value="swimming">Swimming</MenuItem>
                  <MenuItem value="weights">Weight Training</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Duration (minutes)"
                type="number"
                value={formData.activityDuration}
                onChange={handleChange('activityDuration')}
                InputProps={{
                  endAdornment: <FitnessCenter color="action" />
                }}
              />
            </Grid>
          </Grid>
        );
      case 1:
        return (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Calories Consumed"
                type="number"
                value={formData.calories}
                onChange={handleChange('calories')}
                InputProps={{
                  endAdornment: <LocalDining color="action" />
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Water Intake (ml)"
                type="number"
                value={formData.waterIntake}
                onChange={handleChange('waterIntake')}
              />
            </Grid>
          </Grid>
        );
      case 2:
        return (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Sleep Hours"
                type="number"
                value={formData.sleepHours}
                onChange={handleChange('sleepHours')}
                InputProps={{
                  endAdornment: <Bedtime color="action" />
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Sleep Quality</InputLabel>
                <Select
                  value={formData.sleepQuality}
                  onChange={handleChange('sleepQuality')}
                  label="Sleep Quality"
                >
                  <MenuItem value="excellent">Excellent</MenuItem>
                  <MenuItem value="good">Good</MenuItem>
                  <MenuItem value="fair">Fair</MenuItem>
                  <MenuItem value="poor">Poor</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        );
      case 3:
        return (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Glucose Level (mg/dL)"
                type="number"
                value={formData.glucoseLevel}
                onChange={handleChange('glucoseLevel')}
                InputProps={{
                  endAdornment: <MonitorHeart color="action" />
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Blood Pressure (mmHg)"
                value={formData.bloodPressure}
                onChange={handleChange('bloodPressure')}
                placeholder="120/80"
              />
            </Grid>
          </Grid>
        );
      case 4:
        return (
          <Paper elevation={0} sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <CheckCircle color="primary" sx={{ mr: 1 }} /> Review Your Entries
            </Typography>
            <Divider sx={{ my: 2 }} />
            {Object.entries(formData).map(([key, value]) => (
              value && (
                <Typography key={key} sx={{ mb: 1 }}>
                  <strong>
                    {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}:
                  </strong>{' '}
                  {value}
                </Typography>
              )
            ))}
            <TextField
              fullWidth
              multiline
              rows={3}
              label="Additional Notes"
              value={formData.notes}
              onChange={handleChange('notes')}
              sx={{ mt: 2 }}
            />
          </Paper>
        );
      default:
        return 'Unknown step';
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', mb: 3 }}>
        Health Data Tracker
      </Typography>

      {/* Desktop Stepper */}
      <Stepper activeStep={activeStep} sx={{ mb: 4, display: { xs: 'none', md: 'flex' } }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {/* Mobile Stepper */}
      <MobileStepper
        variant="dots"
        steps={steps.length}
        position="static"
        activeStep={activeStep}
        sx={{ mb: 3, display: { md: 'none' } }}
        nextButton={null}
        backButton={null}
      />

      <Paper elevation={3} sx={{ p: 3 }}>
        {getStepContent(activeStep)}

        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
          <Button disabled={activeStep === 0} onClick={handleBack} variant="outlined">
            Back
          </Button>

          {activeStep === steps.length - 1 ? (
            <Button variant="contained" onClick={handleSubmit} color="primary" size="large">
              Submit Data
            </Button>
          ) : (
            <Button variant="contained" onClick={handleNext} color="primary" size="large">
              Next
            </Button>
          )}
        </Box>
      </Paper>
    </Box>
  );
}

export default DataInputForm;