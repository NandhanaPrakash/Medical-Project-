import React, { useState, useEffect } from 'react';
import { 
  Grid, 
  Typography, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem,
  Box,
  Paper,
  Divider,
  ToggleButton,
  ToggleButtonGroup
} from '@mui/material';
import { 
  ShowChart,
  Timeline,
  CalendarToday,
  BarChart
} from '@mui/icons-material';
import ProgressTracker from '../../components/dashboard/ProgressTracker';

function Metrics() {
  const [selectedMetric, setSelectedMetric] = useState('bmi');
  const [timePeriod, setTimePeriod] = useState('week');
  const [chartType, setChartType] = useState('line');
  const [metricData, setMetricData] = useState([]);

  const metrics = [
    { value: 'bmi', label: 'BMI' },
    { value: 'glucose', label: 'Glucose Level' },
    { value: 'sleep', label: 'Sleep Hours' },
    { value: 'activity', label: 'Activity Minutes' }
  ];

  const timePeriods = [
    { value: 'week', label: 'Weekly' },
    { value: 'month', label: 'Monthly' },
    { value: 'year', label: 'Yearly' }
  ];

  useEffect(() => {
    // Simulate data fetching based on selections
    const generateData = () => {
      const data = [];
      const count = timePeriod === 'week' ? 7 : timePeriod === 'month' ? 30 : 12;
      const baseValue = 
        selectedMetric === 'bmi' ? 24.5 : 
        selectedMetric === 'glucose' ? 95 : 
        selectedMetric === 'sleep' ? 7 : 45;
      
      for (let i = 0; i < count; i++) {
        data.push({
          date: timePeriod === 'year' ? 
            `Month ${i+1}` : 
            timePeriod === 'month' ? 
            `Day ${i+1}` : 
            ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][i],
          value: (baseValue + (Math.random() * 2 - 1)).toFixed(1)
        });
      }
      return data;
    };

    setMetricData(generateData());
  }, [selectedMetric, timePeriod]);

  const handleMetricChange = (event) => {
    setSelectedMetric(event.target.value);
  };

  const handleTimePeriodChange = (event, newPeriod) => {
    if (newPeriod) setTimePeriod(newPeriod);
  };

  const handleChartTypeChange = (event, newType) => {
    if (newType) setChartType(newType);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" alignItems="center" mb={3}>
        <ShowChart color="primary" sx={{ fontSize: 32, mr: 1.5 }} />
        <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
          Health Metrics Analytics
        </Typography>
      </Box>

      <Paper elevation={3} sx={{ p: 3, mb: 3, borderRadius: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel id="metric-select-label">Select Metric</InputLabel>
              <Select
                labelId="metric-select-label"
                id="metric-select"
                value={selectedMetric}
                onChange={handleMetricChange}
                label="Select Metric"
              >
                {metrics.map((metric) => (
                  <MenuItem key={metric.value} value={metric.value}>
                    {metric.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <ToggleButtonGroup
              value={timePeriod}
              exclusive
              onChange={handleTimePeriodChange}
              fullWidth
            >
              {timePeriods.map((period) => (
                <ToggleButton key={period.value} value={period.value}>
                  <Box display="flex" alignItems="center">
                    <CalendarToday fontSize="small" sx={{ mr: 1 }} />
                    {period.label}
                  </Box>
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
          </Grid>

          <Grid item xs={12} md={4}>
            <ToggleButtonGroup
              value={chartType}
              exclusive
              onChange={handleChartTypeChange}
              fullWidth
            >
              <ToggleButton value="line">
                <Box display="flex" alignItems="center">
                  <Timeline fontSize="small" sx={{ mr: 1 }} />
                  Line Chart
                </Box>
              </ToggleButton>
              <ToggleButton value="bar">
                <Box display="flex" alignItems="center">
                  <BarChart fontSize="small" sx={{ mr: 1 }} />
                  Bar Chart
                </Box>
              </ToggleButton>
            </ToggleButtonGroup>
          </Grid>
        </Grid>
      </Paper>

      <Paper elevation={3} sx={{ p: 3, borderRadius: 3 }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          {metrics.find(m => m.value === selectedMetric)?.label} Trend
        </Typography>
        <Divider sx={{ mb: 3 }} />
        {metricData.length > 0 ? (
          <ProgressTracker 
            data={metricData} 
            metric={selectedMetric.toUpperCase()} 
            type={chartType}
            height={400}
          />
        ) : (
          <Typography color="textSecondary">
            Loading chart data...
          </Typography>
        )}
      </Paper>
    </Box>
  );
}

export default Metrics;