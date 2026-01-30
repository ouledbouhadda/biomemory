/**
 * AgenticPlanning Component
 * Agentic RAG for experiment planning with Qdrant
 */
import { useState } from 'react';
import { Search, Loader, AlertCircle, CheckCircle } from 'lucide-react';

export default function AgenticPlanning({ onResults }) {
  const [query, setQuery] = useState('');
  const [collection, setCollection] = useState('public_science');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/search/agentic-planning', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({
          query: query.trim(),
          collection: collection
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
      if (onResults) onResults(data);
    } catch (err) {
      setError(err.message || 'Search failed');
      console.error('Agentic search error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2 style={styles.title}>Agentic Experiment Planning</h2>
        <p style={styles.subtitle}>
          AI-powered planning using 9 Qdrant search strategies
        </p>
      </div>

      <form onSubmit={handleSearch} style={styles.form}>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Your Experiment Query</label>
          <textarea
            value={query}
            onChange={(e) => {
              console.log('Query changed:', e.target.value);
              setQuery(e.target.value);
            }}
            placeholder="Describe your experiment... e.g., 'PCR optimization for E.coli at 37°C with Taq polymerase'"
            style={styles.textarea}
            disabled={loading}
          />
        </div>

        <div style={styles.inputGroup}>
          <label style={styles.label}>Collection</label>
          <select
            value={collection}
            onChange={(e) => setCollection(e.target.value)}
            style={styles.select}
            disabled={loading}
          >
            <option value="public_science">Public Science</option>
            <option value="private_experiments">Private Experiments</option>
          </select>
        </div>

        <button
          type="submit"
          disabled={loading || !query.trim()}
          style={{
            ...styles.button,
            opacity: loading || !query.trim() ? 0.6 : 1,
            cursor: loading || !query.trim() ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? (
            <>
              <Loader size={18} style={{ animation: 'spin 1s linear infinite' }} />
              Analyzing with Qdrant...
            </>
          ) : (
            <>
              <Search size={18} />
              Generate Planning
            </>
          )}
        </button>
      </form>

      {error && (
        <div style={styles.errorBox}>
          <AlertCircle size={18} color="#ff6b6b" />
          <span>{error}</span>
        </div>
      )}

      {results && (
        <div style={styles.resultsContainer}>
          {/* Parsed Experiment */}
          <div style={styles.resultSection}>
            <h3 style={styles.resultTitle}>Parsed Experiment</h3>
            <div style={styles.resultContent}>
              <pre style={styles.pre}>
                {JSON.stringify(results.parsed_experiment, null, 2)}
              </pre>
            </div>
          </div>

          {/* Qdrant Insights */}
          {results.qdrant_insights && (
            <div style={styles.resultSection}>
              <h3 style={styles.resultTitle}>Qdrant Insights (9 Search Types)</h3>
              <div style={styles.insightsGrid}>
                {Object.entries(results.qdrant_insights).map(([key, value]) => (
                  <div key={key} style={styles.insightCard}>
                    <div style={styles.insightHeader}>
                      <CheckCircle size={16} color="#4ade80" />
                      <span style={styles.insightTitle}>
                        {key.replace(/_/g, ' ').toUpperCase()}
                      </span>
                    </div>
                    <div style={styles.insightValue}>
                      {Array.isArray(value)
                        ? `${value.length} results found`
                        : typeof value === 'object'
                        ? JSON.stringify(value, null, 2)
                        : String(value)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recommendations */}
          {results.recommendations && (
            <div style={styles.resultSection}>
              <h3 style={styles.resultTitle}>AI Recommendations</h3>
              <div style={styles.resultContent}>
                <p style={styles.recommendationText}>
                  {results.recommendations}
                </p>
              </div>
            </div>
          )}

          {/* Pipeline Time */}
          {results.pipeline_time_ms && (
            <div style={styles.pipelineTime}>
              Executed in {results.pipeline_time_ms.toFixed(2)}ms
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

const styles = {
  container: {
    padding: '2rem',
    maxWidth: '1200px',
    margin: '0 auto',
  },
  header: {
    marginBottom: '2rem',
  },
  title: {
    fontSize: '2rem',
    fontWeight: 'bold',
    color: '#0a1131',
    marginBottom: '0.5rem',
  },
  subtitle: {
    fontSize: '0.95rem',
    color: '#6b7280',
  },
  form: {
    display: 'grid',
    gap: '1.5rem',
    marginBottom: '2rem',
    background: '#f9fafb',
    padding: '2rem',
    borderRadius: '12px',
    border: '1px solid #e5e7eb',
  },
  inputGroup: {
    display: 'grid',
    gap: '0.5rem',
  },
  label: {
    fontSize: '0.9rem',
    fontWeight: '500',
    color: '#374151',
  },
  textarea: {
    padding: '1rem',
    borderRadius: '8px',
    border: '1px solid #d1d5db',
    background: '#fff',
    color: '#1f2937',
    fontSize: '0.95rem',
    fontFamily: 'inherit',
    minHeight: '120px',
    resize: 'vertical',
    width: '100%',
    boxSizing: 'border-box',
  },
  select: {
    padding: '0.75rem',
    borderRadius: '8px',
    border: '1px solid #d1d5db',
    background: '#fff',
    color: '#1f2937',
    fontSize: '0.95rem',
  },
  button: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.75rem',
    padding: '1rem 2rem',
    borderRadius: '8px',
    border: 'none',
    background: 'linear-gradient(135deg, #0a1131 0%, #1e3a5f 100%)',
    color: '#fff',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
  },
  errorBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    padding: '1rem',
    borderRadius: '8px',
    background: '#fef2f2',
    border: '1px solid #fecaca',
    color: '#991b1b',
    marginBottom: '2rem',
  },
  resultsContainer: {
    display: 'grid',
    gap: '2rem',
  },
  resultSection: {
    background: '#f9fafb',
    padding: '1.5rem',
    borderRadius: '12px',
    border: '1px solid #e5e7eb',
  },
  resultTitle: {
    fontSize: '1.2rem',
    fontWeight: '600',
    color: '#0a1131',
    marginBottom: '1rem',
  },
  resultContent: {
    color: '#374151',
    lineHeight: '1.6',
  },
  pre: {
    background: '#1f2937',
    padding: '1rem',
    borderRadius: '6px',
    overflow: 'auto',
    fontSize: '0.85rem',
    color: '#4ade80',
    margin: 0,
  },
  insightsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '1rem',
  },
  insightCard: {
    background: '#fff',
    padding: '1rem',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
    boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
  },
  insightHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '0.5rem',
  },
  insightTitle: {
    fontSize: '0.85rem',
    fontWeight: '600',
    color: '#b45309',
  },
  insightValue: {
    fontSize: '0.9rem',
    color: '#4b5563',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  recommendationText: {
    fontSize: '0.95rem',
    lineHeight: '1.7',
    color: '#374151',
  },
  pipelineTime: {
    textAlign: 'center',
    fontSize: '0.85rem',
    color: '#059669',
    padding: '0.75rem',
    background: '#f0fdf4',
    borderRadius: '8px',
  },
};
