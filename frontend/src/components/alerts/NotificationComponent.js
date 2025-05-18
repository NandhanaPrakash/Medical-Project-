import React, { useState, useEffect } from 'react';
import { Snackbar, Alert } from '@mui/material';

function NotificationComponent({ open, onClose, message, severity }) {
  return (
    <Snackbar open={open} autoHideDuration={6000} onClose={onClose} anchorOrigin={{ vertical: 'top', horizontal: 'right' }}>
      <Alert onClose={onClose} severity={severity} sx={{ width: '100%' }}>
        {message}
      </Alert>
    </Snackbar>
  );
}

export default NotificationComponent;