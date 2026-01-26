/**
 * Main Entry Point
 * Renders the React application
 */
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Global styles
const globalStyles = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  input:focus,
  textarea:focus,
  button:focus {
    outline: 2px solid #0a1131;
    outline-offset: 2px;
  }

  button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(23, 36, 94, 0.3);
  }

  button:active {
    transform: translateY(0);
  }

  button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none !important;
  }

  .card:hover {
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
  }

  * {
    scrollbar-width: thin;
    scrollbar-color: #0a1131 #f1f1f1;
  }

  *::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  *::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
  }

  *::-webkit-scrollbar-thumb {
    background: #0a1131;
    border-radius: 10px;
  }

  *::-webkit-scrollbar-thumb:hover {
    background: #0a1131;
  }
`;

// Inject global styles
const styleSheet = document.createElement('style');
styleSheet.textContent = globalStyles;
document.head.appendChild(styleSheet);

// Render app
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
