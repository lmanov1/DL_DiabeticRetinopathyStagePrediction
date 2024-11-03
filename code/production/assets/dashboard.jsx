import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadialBarChart, RadialBar } from 'recharts';
import { Card, CardContent } from '@/components/ui/card';

const MetricsDashboard = () => {
  // Overall Metrics Data
  const overallMetrics = [
    { name: 'Accuracy', value: 82.32, fill: '#0073CF' },
    { name: 'AUC', value: 86.04, fill: '#00B5A5' },
    { name: 'Macro Precision', value: 69.22, fill: '#FF7F50' },
    { name: 'Macro Recall', value: 47.32, fill: '#FFB347' },
    { name: 'Macro F1', value: 50.40, fill: '#87CEEB' }
  ];

  // Per-Class Metrics
  const classMetrics = [
    {
      class: 'Class 0',
      precision: 86,
      recall: 98,
      f1: 91,
    },
    {
      class: 'Class 1',
      precision: 67,
      recall: 1,
      f1: 1,
    },
    {
      class: 'Class 2',
      precision: 66,
      recall: 61,
      f1: 63,
    },
    {
      class: 'Class 3',
      precision: 63,
      recall: 34,
      f1: 44,
    },
    {
      class: 'Class 4',
      precision: 65,
      recall: 43,
      f1: 52,
    }
  ];

  return (
    <div className="space-y-8">
      {/* Overall Metrics Visualization */}
      <Card className="p-6">
        <h3 className="text-xl font-bold mb-4">Overall Model Performance</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <RadialBarChart 
              cx="50%" 
              cy="50%" 
              innerRadius="20%" 
              outerRadius="100%" 
              data={overallMetrics} 
              startAngle={180} 
              endAngle={0}
            >
              <RadialBar
                minAngle={15}
                background
                clockWise={true}
                dataKey="value"
                label={{ fill: '#666', position: 'insideStart' }}
              />
              <Legend />
              <Tooltip />
            </RadialBarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Per-Class Performance */}
      <Card className="p-6">
        <h3 className="text-xl font-bold mb-4">Per-Class Performance Metrics</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart 
              data={classMetrics}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="class" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="precision" fill="#0073CF" name="Precision" />
              <Bar dataKey="recall" fill="#00B5A5" name="Recall" />
              <Bar dataKey="f1" fill="#FF7F50" name="F1 Score" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Placeholder for Confusion Matrix Visualization */}
      <Card className="p-6">
        <h3 className="text-xl font-bold mb-4">Confusion Matrix</h3>
        <div className="grid grid-cols-5 gap-1">
          {[
            [2523, 0, 59, 0, 3],
            [233, 2, 34, 0, 0],
            [173, 1, 307, 13, 9],
            [3, 0, 50, 29, 4],
            [17, 0, 18, 4, 30]
          ].map((row, i) => (
            row.map((cell, j) => (
              <div 
                key={`${i}-${j}`}
                className="p-2 text-center text-sm"
                style={{
                  backgroundColor: `rgba(0, 115, 207, ${cell/2523})`,
                  color: cell/2523 > 0.5 ? 'white' : 'black'
                }}
              >
                {cell}
              </div>
            ))
          ))}
        </div>
      </Card>

      {/* Placeholder for Challenges Section */}
      <Card className="p-6">
        <h3 className="text-xl font-bold mb-4">Development Challenges</h3>
        <div className="space-y-4">
          <div className="p-4 bg-gray-50 rounded">
            <h4 className="font-bold text-lg mb-2">Challenge 1: [Title]</h4>
            <p className="text-gray-700">Description of the challenge and how it was overcome...</p>
          </div>
          <div className="p-4 bg-gray-50 rounded">
            <h4 className="font-bold text-lg mb-2">Challenge 2: [Title]</h4>
            <p className="text-gray-700">Description of the challenge and how it was overcome...</p>
          </div>
          <div className="p-4 bg-gray-50 rounded">
            <h4 className="font-bold text-lg mb-2">Challenge 3: [Title]</h4>
            <p className="text-gray-700">Description of the challenge and how it was overcome...</p>
          </div>
        </div>
      </Card>

      {/* Placeholder for Hugging Face Interface Screenshots */}
      <Card className="p-6">
        <h3 className="text-xl font-bold mb-4">Hugging Face Interface</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="border rounded p-4">
            <h4 className="font-bold mb-2">Upload Interface</h4>
            <img src="/api/placeholder/400/300" alt="Upload Interface Screenshot" className="w-full rounded" />
          </div>
          <div className="border rounded p-4">
            <h4 className="font-bold mb-2">Results Display</h4>
            <img src="/api/placeholder/400/300" alt="Results Interface Screenshot" className="w-full rounded" />
          </div>
        </div>
      </Card>
    </div>
  );
};

export default MetricsDashboard;
