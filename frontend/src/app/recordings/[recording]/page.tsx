"use client";

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface Scores {
  overall: number[];
  clarity: number[];
  pace: number[];
  volume: number[];
  posture: number[];
  expression: number[];
  eye_contact: number[];
  speech: number[];
  engagement: number[];
}

interface RecordingData {
  timestamp: string;
  duration_seconds: number;
  scores: Scores;
  video_file: string;
}

interface Recording {
  name: string;
  date: string;
  videoPath: string;
  jsonPath: string;
}

export default function RecordingPage() {
  const router = useRouter();
  const params = useParams();
  const [recordingData, setRecordingData] = useState<RecordingData | null>(null);
  const [recording, setRecording] = useState<Recording | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRecording = async () => {
      try {
        setError(null);
        const response = await fetch('http://localhost:8000/recordings');
        const recordings: Recording[] = await response.json();
        const found = recordings.find((r: any) => r.videoPath.includes(params.recording as string));
        if (found) {
          setRecording(found);
          if (found.jsonPath) {
            const dataResponse = await fetch(`http://localhost:8000${found.jsonPath}`);
            if (!dataResponse.ok) {
              throw new Error(`Failed to fetch recording data: ${dataResponse.statusText}`);
            }
            const data = await dataResponse.json();
            if (!data.scores) {
              throw new Error('Recording data is missing scores');
            }
            setRecordingData(data);
          } else {
            throw new Error('Recording JSON data not found');
          }
        } else {
          throw new Error('Recording not found');
        }
      } catch (error) {
        console.error('Error fetching recording:', error);
        setError(error instanceof Error ? error.message : 'An error occurred');
      }
    };

    fetchRecording();
  }, [params.recording]);

  if (!recording || !recordingData) {
    return (
      <div className="p-8">
        <div className="max-w-4xl mx-auto">
          <button
            onClick={() => router.push('/')}
            className="mb-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          >
            ← Back to Recordings
          </button>
          <div className="text-center py-8 text-gray-500">
            {error ? (
              <div className="text-red-500">{error}</div>
            ) : (
              'Loading recording...'
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">Recording Details</h1>
          <button
            onClick={() => router.push('/')}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            ← Back to List
          </button>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-4 text-gray-800">
          <video 
            controls 
            className="w-full rounded"
            src={`http://localhost:8000${recording.videoPath}`}
            onError={(e) => console.error('Video playback error:', e)}
          />
          <h2 className="mt-4 font-semibold">{recording.name}</h2>
          <p className="text-gray-500">{recording.date}</p>
          
          <div className="mt-8">
            <h3 className="text-lg font-semibold mb-4 text-gray-800">Performance Metrics</h3>
            <div className="space-y-6">
              <div className="mb-8">
                <h4 className="text-lg font-semibold mb-4 text-gray-800">Overall Performance</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart
                    data={recordingData.scores.overall.map((value, index) => ({
                      time: index,
                      overall: value,
                    }))}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -5 }} />
                    <YAxis domain={[0, 100]} label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="overall" stroke="#8884d8" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div>
                <h4 className="text-lg font-semibold mb-4 text-gray-800">Individual Metrics</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart
                    data={recordingData.scores.overall.map((value, index) => ({
                      time: index,
                      clarity: recordingData.scores.clarity[index],
                      pace: recordingData.scores.pace[index],
                      volume: recordingData.scores.volume[index],
                      posture: recordingData.scores.posture[index],
                      expression: recordingData.scores.expression[index],
                      eye_contact: recordingData.scores.eye_contact[index],
                      speech: recordingData.scores.speech[index],
                      engagement: recordingData.scores.engagement[index],
                    }))}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -5 }} />
                    <YAxis domain={[0, 100]} label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="clarity" stroke="#82ca9d" />
                    <Line type="monotone" dataKey="pace" stroke="#ffc658" />
                    <Line type="monotone" dataKey="volume" stroke="#ff7300" />
                    <Line type="monotone" dataKey="posture" stroke="#8884d8" />
                    <Line type="monotone" dataKey="expression" stroke="#2196f3" />
                    <Line type="monotone" dataKey="eye_contact" stroke="#673ab7" />
                    <Line type="monotone" dataKey="speech" stroke="#e91e63" />
                    <Line type="monotone" dataKey="engagement" stroke="#ff0000" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="mt-8 p-6 bg-gray-50 rounded-lg">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Performance Analysis</h3>
                {(() => {
                  // Calculate average scores for each metric
                  const averages = Object.entries(recordingData.scores).reduce((acc, [key, values]) => {
                    if (Array.isArray(values)) {
                      acc[key] = values.reduce((sum, val) => sum + val, 0) / values.length;
                    }
                    return acc;
                  }, {} as Record<string, number>);

                  // Sort metrics by score
                  const sortedMetrics = Object.entries(averages)
                    .filter(([key]) => key !== 'overall')
                    .sort((a, b) => b[1] - a[1]);

                  const strengths = sortedMetrics.slice(0, 3);
                  const improvements = sortedMetrics.slice(-3).reverse();

                  const formatMetric = (metric: string) => {
                    return metric
                      .split('_')
                      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                      .join(' ');
                  };

                  return (
                    <div className="space-y-6">
                      <div>
                        <h4 className="text-md font-semibold text-green-700 mb-2">Top Strengths</h4>
                        <ul className="space-y-2">
                          {strengths.map(([metric, score]) => (
                            <li key={metric} className="flex items-center justify-between">
                              <span>{formatMetric(metric)}</span>
                              <span className="font-medium text-green-600">{score.toFixed(1)}%</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h4 className="text-md font-semibold text-amber-700 mb-2">Areas for Improvement</h4>
                        <ul className="space-y-2">
                          {improvements.map(([metric, score]) => (
                            <li key={metric} className="flex items-center justify-between">
                              <span>{formatMetric(metric)}</span>
                              <span className="font-medium text-amber-600">{score.toFixed(1)}%</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div className="text-sm text-gray-600 mt-4">
                        <p className="mb-2">
                          <strong>Overall Performance:</strong> {averages.overall.toFixed(1)}%
                        </p>
                        <p>
                          This analysis is based on average scores throughout your presentation.
                          Focus on maintaining your strengths while working to improve the areas with lower scores.
                        </p>
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
