import React from 'react';
import { Link } from 'react-router-dom';
import { 
  AppBar, 
  Toolbar, 
  Button,
  IconButton
} from '@mui/material';
import {
  Chat as ChatIcon
} from '@mui/icons-material';

// Main Navbar component
function Navbar() {
  const openChatbot = () => {
    window.open('http://localhost:8000', '_blank'); // Opens in new tab
  };

  return (
    <AppBar position="static" sx={{ mb: 3 }}>
      <Toolbar>
        <Button 
          color="inherit" 
          component={Link} 
          to="/patient/dashboard"
          sx={{ textTransform: 'none' }}
        >
          Dashboard
        </Button>
        <Button 
          color="inherit" 
          component={Link} 
          to="/patient/data-input"
          sx={{ textTransform: 'none' }}
        >
          Data Input
        </Button>
        <Button 
          color="inherit" 
          component={Link} 
          to="/patient/recommendations"
          sx={{ textTransform: 'none' }}
        >
          Recommendations
        </Button>
        <Button 
          color="inherit" 
          component={Link} 
          to="/patient/alerts"
          sx={{ textTransform: 'none' }}
        >
          Alerts
        </Button>
        <Button 
          color="inherit" 
          component={Link} 
          to="/patient/metrics"
          sx={{ textTransform: 'none' }}
        >
          Metrics
        </Button>
        <Button 
          color="inherit" 
          component={Link} 
          to="/patient/feedback"
          sx={{ textTransform: 'none' }}
        >
          Feedback
        </Button>

        {/* Chatbot Redirect Button */}
        <IconButton 
          color="inherit"
          onClick={openChatbot}
          sx={{ ml: 'auto' }}
          aria-label="Open Chatbot"
        >
          <ChatIcon />
        </IconButton>
      </Toolbar>
    </AppBar>
  );
}

export default Navbar;
