import { useState, useEffect } from 'react';

function App() {
  const [backendMessage, setBackendMessage] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    const fetchBackendData = async () => {
      try {
        // Use environment variable for backend URL, fallback to localhost for development
        const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';
        const response = await fetch(`${backendUrl}/api/hello`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        setBackendMessage(data.message);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch from backend');
      } finally {
        setLoading(false);
      }
    };

    fetchBackendData();
  }, []);

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold text-center mb-4">RAG Pipeline Frontend</h1>
      
      <div className="max-w-md mx-auto bg-white rounded-lg shadow-md p-6">
        <h2 className="text-lg font-semibold mb-2">Backend Connection Status:</h2>
        
        {loading && (
          <p className="text-blue-600">Connecting to backend...</p>
        )}
        
        {error && (
          <p className="text-red-600">Error: {error}</p>
        )}
        
        {backendMessage && (
          <p className="text-green-600">✓ {backendMessage}</p>
        )}
      </div>
    </div>
  );
}

export default App;
