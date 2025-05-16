import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, LineElement, PointElement, LinearScale, Title, CategoryScale } from 'chart.js';
import { Card, CardContent, Typography } from '@mui/material';

ChartJS.register(LineElement, PointElement, LinearScale, Title, CategoryScale);

function ProgressTracker({ data, metric }) {
  const chartData = {
    labels: data.map((item) => item.date), // Example: ['2025-04-01', '2025-04-08', ...]
    datasets: [
      {
        label: metric,
        data: data.map((item) => item.value), // Example: [24.2, 24.8, ...]
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: `Progress of ${metric} Over Time`,
      },
    },
    scales: {
      x: {
        type: 'category',
        labels: chartData.labels,
      },
      y: {
        type: 'linear',
        beginAtZero: true,
      },
    },
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {metric} Progress
        </Typography>
        <Line data={chartData} options={chartOptions} />
      </CardContent>
    </Card>
  );
}

export default ProgressTracker;