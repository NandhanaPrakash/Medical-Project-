import React, { useState } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  List, 
  ListItem, 
  ListItemText, 
  Button, 
  Divider,
  Box,
  Paper,
  Chip,
  Avatar,
  ListItemAvatar
} from '@mui/material';
import { 
  LocalDining,
  DirectionsRun,
  Bedtime,
  Spa,
  AddCircleOutline
} from '@mui/icons-material';

function Recommendations() {
  const [recommendations, setRecommendations] = useState([
    { 
      id: 1, 
      type: 'Nutrition', 
      text: 'Increase intake of leafy greens and colorful vegetables to at least 3 servings per day.', 
      priority: 'High',
      icon: <LocalDining color="primary" />
    },
    { 
      id: 2, 
      type: 'Exercise', 
      text: 'Incorporate 30 minutes of brisk walking or cycling at least 5 days a week.', 
      priority: 'Medium',
      icon: <DirectionsRun color="primary" />
    },
    { 
      id: 3, 
      type: 'Sleep', 
      text: 'Establish a consistent bedtime routine aiming for 7-8 hours of quality sleep.', 
      priority: 'High',
      icon: <Bedtime color="primary" />
    },
    { 
      id: 4, 
      type: 'Stress', 
      text: 'Practice 10 minutes of mindfulness meditation daily to reduce stress levels.', 
      priority: 'Medium',
      icon: <Spa color="primary" />
    }
  ]);

  const handleAccept = (id) => {
    setRecommendations(recs => recs.map(r => 
      r.id === id ? {...r, status: 'Accepted'} : r
    ));
  };

  const handleModify = (id) => {
    setRecommendations(recs => recs.map(r => 
      r.id === id ? {...r, status: 'Modification Requested'} : r
    ));
  };

  const handleRequestNew = () => {
    // Simulate adding a new recommendation
    const newRec = {
      id: recommendations.length + 1,
      type: 'General',
      text: 'Stay hydrated by drinking at least 8 glasses of water daily.',
      priority: 'Low',
      icon: <Spa color="primary" />,
      status: 'New'
    };
    setRecommendations([...recommendations, newRec]);
  };

  const getPriorityColor = (priority) => {
    switch(priority) {
      case 'High': return 'error';
      case 'Medium': return 'warning';
      case 'Low': return 'success';
      default: return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" alignItems="center" mb={3}>
        <Typography variant="h4" sx={{ 
          fontWeight: 600,
          color: 'primary.main'
        }}>
          Personalized Recommendations
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddCircleOutline />}
          onClick={handleRequestNew}
          sx={{ ml: 'auto' }}
        >
          New Recommendation
        </Button>
      </Box>

      <Paper elevation={3} sx={{ borderRadius: 3, overflow: 'hidden' }}>
        <List sx={{ p: 0 }}>
          {recommendations.map((recommendation, index) => (
            <React.Fragment key={recommendation.id}>
              <ListItem 
                alignItems="flex-start"
                sx={{ 
                  p: 3,
                  bgcolor: recommendation.status === 'Accepted' ? 'action.selected' : 'background.paper',
                  transition: 'background-color 0.3s',
                  '&:hover': { bgcolor: 'action.hover' }
                }}
              >
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: 'primary.light' }}>
                    {recommendation.icon}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={
                    <Box display="flex" alignItems="center">
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {recommendation.type}
                      </Typography>
                      <Chip
                        label={recommendation.priority}
                        color={getPriorityColor(recommendation.priority)}
                        size="small"
                        sx={{ ml: 2 }}
                      />
                      {recommendation.status && (
                        <Chip
                          label={recommendation.status}
                          variant="outlined"
                          size="small"
                          sx={{ ml: 1 }}
                        />
                      )}
                    </Box>
                  }
                  secondary={
                    <React.Fragment>
                      <Typography
                        component="span"
                        variant="body1"
                        color="text.primary"
                        display="block"
                        sx={{ mt: 1 }}
                      >
                        {recommendation.text}
                      </Typography>
                      <Box sx={{ mt: 2 }}>
                        <Button 
                          variant="contained" 
                          size="small" 
                          onClick={() => handleAccept(recommendation.id)}
                          sx={{ mr: 1 }}
                        >
                          Accept
                        </Button>
                        <Button 
                          variant="outlined" 
                          size="small" 
                          onClick={() => handleModify(recommendation.id)}
                        >
                          Request Modification
                        </Button>
                      </Box>
                    </React.Fragment>
                  }
                />
              </ListItem>
              {index < recommendations.length - 1 && (
                <Divider variant="inset" component="li" />
              )}
            </React.Fragment>
          ))}
        </List>
      </Paper>
    </Box>
  );
}

export default Recommendations;