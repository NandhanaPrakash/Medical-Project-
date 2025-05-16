import React, { useState, useEffect } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box,
  Chip,
  IconButton,
  Tooltip,
  TablePagination,
  useTheme,
  Skeleton
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
  FilterList as FilterListIcon
} from '@mui/icons-material';

const severityColors = {
  Critical: 'error',
  High: 'warning',
  Medium: 'info',
  Low: 'success'
};

const severityIcons = {
  Critical: <ErrorIcon />,
  High: <WarningIcon />,
  Medium: <InfoIcon />,
  Low: <InfoIcon />
};

function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);
  const theme = useTheme();

  useEffect(() => {
    // Simulate API fetch
    const fetchAlerts = async () => {
      setLoading(true);
      try {
        // Replace with actual API call
        await new Promise(resolve => setTimeout(resolve, 800));
        const mockData = [
          { id: 1, metric: 'Glucose', value: 200, timestamp: '2025-04-10 10:00', severity: 'High' },
          { id: 2, metric: 'Blood Pressure', value: '150/95', timestamp: '2025-04-09 08:30', severity: 'Critical' },
          { id: 3, metric: 'Heart Rate', value: 110, timestamp: '2025-04-08 14:30', severity: 'Medium' },
          { id: 4, metric: 'BMI', value: 31.5, timestamp: '2025-04-07 16:45', severity: 'High' },
          { id: 5, metric: 'Oxygen Saturation', value: 88, timestamp: '2025-04-06 22:15', severity: 'Critical' },
          { id: 6, metric: 'Sleep Duration', value: 4.2, timestamp: '2025-04-05 07:00', severity: 'Low' },
          { id: 7, metric: 'Cholesterol', value: 240, timestamp: '2025-04-04 12:30', severity: 'High' }
        ];
        setAlerts(mockData);
      } catch (error) {
        console.error('Failed to fetch alerts:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchAlerts();
  }, []);

  const handleRefresh = () => {
    // Trigger refresh
    setLoading(true);
    setTimeout(() => setLoading(false), 800);
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        mb: 3 
      }}>
        <Typography variant="h5" component="h2" sx={{ fontWeight: 600 }}>
          Health Alerts
        </Typography>
        <Box>
          <Tooltip title="Refresh">
            <IconButton onClick={handleRefresh} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Filter">
            <IconButton disabled={loading}>
              <FilterListIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Paper elevation={3} sx={{ borderRadius: 2 }}>
        <TableContainer>
          <Table aria-label="health alerts table">
            <TableHead sx={{ bgcolor: theme.palette.grey[100] }}>
              <TableRow>
                <TableCell sx={{ fontWeight: 600 }}>Metric</TableCell>
                <TableCell align="right" sx={{ fontWeight: 600 }}>Value</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Timestamp</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Severity</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {loading ? (
                Array(rowsPerPage).fill(0).map((_, index) => (
                  <TableRow key={`skeleton-${index}`}>
                    <TableCell><Skeleton animation="wave" /></TableCell>
                    <TableCell><Skeleton animation="wave" /></TableCell>
                    <TableCell><Skeleton animation="wave" /></TableCell>
                    <TableCell><Skeleton animation="wave" /></TableCell>
                  </TableRow>
                ))
              ) : (
                alerts
                  .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                  .map((alert) => (
                    <TableRow 
                      key={alert.id}
                      hover
                      sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                    >
                      <TableCell component="th" scope="row">
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {severityIcons[alert.severity]}
                          <Box sx={{ ml: 1.5 }}>{alert.metric}</Box>
                        </Box>
                      </TableCell>
                      <TableCell align="right">{alert.value}</TableCell>
                      <TableCell>{alert.timestamp}</TableCell>
                      <TableCell>
                        <Chip 
                          label={alert.severity} 
                          color={severityColors[alert.severity] || 'default'}
                          variant="outlined"
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={alerts.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
          sx={{ borderTop: `1px solid ${theme.palette.divider}` }}
        />
      </Paper>
    </Box>
  );
}

export default Alerts;