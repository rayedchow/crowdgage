"use client";

import { useState, useEffect } from 'react';
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

export default function RecordingsPage() {
  const [view, setView] = useState<'list' | 'details'>('list');
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [selectedRecording, setSelectedRecording] = useState<Recording | null>(null);
  const [recordingData, setRecordingData] = useState<RecordingData | null>(null);

  const fetchRecordingData = async (jsonPath: string) => {
    try {
      const response = await fetch(`http://localhost:8000${jsonPath}`);
      const data = await response.json();
      setRecordingData(data);
    } catch (error) {
      console.error('Error fetching recording data:', error);
    }
  };

  useEffect(() => {
    const fetchRecordings = async () => {
      try {
        const response = await fetch('http://localhost:8000/recordings');
        const data = await response.json();
        setRecordings(data);
      } catch (error) {
        console.error('Error fetching recordings:', error);
      }
    };

    fetchRecordings();
  }, []);

  return (
    <div className="p-8">
      <main className="max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">Presentation Recordings</h1>
          {view === 'details' && (
            <button
              onClick={() => {
                setView('list');
                setSelectedRecording(null);
                setRecordingData(null);
              }}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
            >
              ← Back to List
            </button>
          )}
        </div>

        {view === 'list' ? (
          <div className="bg-white rounded-lg shadow-lg p-4">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Available Recordings</h2>
            <div className="space-y-2">
              {recordings.map((recording) => (
                <button
                  key={recording.videoPath}
                  onClick={() => {
                    setSelectedRecording(recording);
                    if (recording.jsonPath) {
                      fetchRecordingData(recording.jsonPath);
                      setView('details');
                    }
                  }}
                  className={`w-full text-left p-3 rounded ${selectedRecording?.videoPath === recording.videoPath
                    ? 'bg-blue-900 text-white'
                    : 'bg-gray-100 hover:bg-gray-200'}`}
                >
                  <div className="font-medium text-gray-800">{recording.name}</div>
                  <div className="text-sm text-gray-500">{recording.date}</div>
                </button>
              ))}
              {recordings.length === 0 && (
                <p className="text-gray-500 text-center py-4">
                  No recordings found
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow-lg p-4 text-gray-800">
            {selectedRecording && recordingData && (
              <div>
                <video 
                  controls 
                  className="w-full rounded"
                  src={`http://localhost:8000${selectedRecording.videoPath}`}
                  onError={(e) => console.error('Video playback error:', e)}
                />
                <h2 className="mt-4 font-semibold">{selectedRecording.name}</h2>
                <p className="text-gray-500">{selectedRecording.date}</p>
                {recordingData && (
                  <div className="mt-8">
                    <h3 className="text-lg font-semibold mb-4 text-gray-800">Performance Metrics</h3>
                    <div className="space-y-6">
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart
                          data={recordingData.scores.overall.map((value, index) => ({
                            time: index,
                            overall: value,
                            clarity: recordingData.scores.clarity[index],
                            pace: recordingData.scores.pace[index],
                            volume: recordingData.scores.volume[index],
                          }))}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="time" label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -5 }} />
                          <YAxis domain={[0, 100]} label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="overall" stroke="#8884d8" />
                          <Line type="monotone" dataKey="clarity" stroke="#82ca9d" />
                          <Line type="monotone" dataKey="pace" stroke="#ffc658" />
                          <Line type="monotone" dataKey="volume" stroke="#ff7300" />
                        </LineChart>
                      </ResponsiveContainer>

                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart
                          data={recordingData.scores.overall.map((value, index) => ({
                            time: index,
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
                          <Line type="monotone" dataKey="posture" stroke="#8884d8" />
                          <Line type="monotone" dataKey="expression" stroke="#82ca9d" />
                          <Line type="monotone" dataKey="eye_contact" stroke="#ffc658" />
                          <Line type="monotone" dataKey="speech" stroke="#ff7300" />
                          <Line type="monotone" dataKey="engagement" stroke="#ff0000" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
