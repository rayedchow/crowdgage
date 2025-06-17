"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

interface Recording {
  name: string;
  date: string;
  videoPath: string;
  jsonPath: string;
}

export default function Home() {
  const router = useRouter();
  const [recordings, setRecordings] = useState<Recording[]>([]);

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
        </div>

        <div className="bg-white rounded-lg shadow-lg p-4">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Available Recordings</h2>
          <div className="space-y-2">
            {recordings.map((recording) => {
              const recordingId = recording.videoPath.split('/').pop()?.replace('.mp4', '');
              return (
                <button
                  key={recording.videoPath}
                  onClick={() => router.push(`/recordings/${recordingId}`)}
                  className="w-full text-left p-3 rounded bg-gray-100 hover:bg-gray-200"
                >
                  <div className="font-medium text-gray-800">{recording.name}</div>
                  <div className="text-sm text-gray-500">{recording.date}</div>
                </button>
              );
            })}
            {recordings.length === 0 && (
              <p className="text-gray-500 text-center py-4">
                No recordings found
              </p>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
