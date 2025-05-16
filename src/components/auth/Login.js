import React, { useState } from 'react';
import { 
  TextField, 
  Button, 
  Grid, 
  Typography, 
  Paper, 
  Alert, 
  Box, 
  IconButton, 
  InputAdornment,
  Link,
  Divider
} from '@mui/material';
import { 
  Visibility, 
  VisibilityOff,
  Lock,
  Person,
  ArrowBack
} from '@mui/icons-material';
import { useAuth } from '../../authContext';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setIsSubmitting(true);
    
    try {
      await login({ username, password });
      // Redirect is handled in authContext on successful login
    } catch (err) {
      setError(err.message || 'Login failed. Please check your credentials and try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClickShowPassword = () => {
    setShowPassword(!showPassword);
  };

  const handleMouseDownPassword = (event) => {
    event.preventDefault();
  };

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        p: 2
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Paper
          elevation={6}
          sx={{
            padding: 4,
            width: '100%',
            maxWidth: 450,
            borderRadius: 3,
            position: 'relative'
          }}
        >
          <IconButton 
            sx={{ position: 'absolute', left: 16, top: 16 }}
            onClick={() => navigate(-1)}
          >
            <ArrowBack />
          </IconButton>

          <Box sx={{ textAlign: 'center', mb: 4 }}>
            <Lock color="primary" sx={{ fontSize: 50 }} />
            <Typography variant="h4" component="h1" sx={{ mt: 1, fontWeight: 600 }}>
              Welcome Back
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Sign in to continue to your health dashboard
            </Typography>
          </Box>

          <form onSubmit={handleSubmit} noValidate>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <TextField
                  label="Username or Email"
                  fullWidth
                  required
                  autoFocus
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Person color="action" />
                      </InputAdornment>
                    ),
                  }}
                  variant="outlined"
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Password"
                  type={showPassword ? 'text' : 'password'}
                  fullWidth
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Lock color="action" />
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          aria-label="toggle password visibility"
                          onClick={handleClickShowPassword}
                          onMouseDown={handleMouseDownPassword}
                          edge="end"
                        >
                          {showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                  variant="outlined"
                />
              </Grid>
              {error && (
                <Grid item xs={12}>
                  <Alert severity="error" sx={{ width: '100%' }}>
                    {error}
                  </Alert>
                </Grid>
              )}
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <Link href="/forgot-password" variant="body2" underline="hover">
                    Forgot password?
                  </Link>
                </Box>
              </Grid>
              <Grid item xs={12}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  fullWidth
                  size="large"
                  disabled={isSubmitting}
                  sx={{ py: 1.5 }}
                >
                  {isSubmitting ? 'Signing In...' : 'Sign In'}
                </Button>
              </Grid>
              <Grid item xs={12}>
                <Divider sx={{ my: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    OR
                  </Typography>
                </Divider>
              </Grid>
              <Grid item xs={12}>
                <Button
                  variant="outlined"
                  color="primary"
                  fullWidth
                  size="large"
                  onClick={() => navigate('/register')}
                  sx={{ py: 1.5 }}
                >
                  Create New Account
                </Button>
              </Grid>
            </Grid>
          </form>
        </Paper>
      </motion.div>
    </Box>
  );
}

export default Login;