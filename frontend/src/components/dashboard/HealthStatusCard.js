import React from 'react';
import { Card, CardContent, Typography, Box, useTheme } from '@mui/material';

const HealthStatusCard = ({ title, value, unit, icon, status }) => {
  const theme = useTheme();
  
  const getStatusColor = () => {
    if (status.includes('Normal') || status.includes('Good')) return theme.palette.success.main;
    if (status.includes('High') || status.includes('Monitor')) return theme.palette.warning.main;
    if (status.includes('Needs')) return theme.palette.error.main;
    return theme.palette.primary.main;
  };

  return (
    <Card sx={{ 
      height: '100%',
      borderRadius: 3,
      boxShadow: 3,
      transition: 'transform 0.3s',
      '&:hover': { transform: 'scale(1.02)' }
    }}>
      <CardContent sx={{ p: 2 }}>
        <Box display="flex" alignItems="center" mb={1}>
          {icon}
          <Typography variant="h6" sx={{ ml: 1.5, fontWeight: 600 }}>
            {title}
          </Typography>
        </Box>
        
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
          {value} {unit}
        </Typography>
        
        <Box sx={{ 
          display: 'inline-block',
          px: 1.5,
          py: 0.5,
          borderRadius: 2,
          bgcolor: getStatusColor() + '20',
          color: getStatusColor()
        }}>
          {status}
        </Box>
      </CardContent>
    </Card>
  );
};

export default HealthStatusCard;
