import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

test('renders title', async () => {
  render(<App />);
  
  expect(screen.getByText(/RAG Pipeline Frontend/i)).toBeInTheDocument();
  
  // Wait for the backend message to appear (or error message)
  // await waitFor(() => {
  //   const successMessage = screen.queryByText(/Hello from Flask backend/i);
  //   const errorMessage = screen.queryByText(/Error:/i);
  //   expect(successMessage || errorMessage).toBeInTheDocument();
  // }, { timeout: 5000 });
});
