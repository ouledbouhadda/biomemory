/**
 * Dashboard Component
 * Main interface for BIOMEMORY
 */
import { useState, useEffect } from 'react';
import SearchPanel from './SearchPanel';
import ExperimentUpload from './ExperimentUpload';
import SimilarExperiments from './SimilarExperiments';
import DesignSuggestions from './DesignSuggestions';
import ReproducibilityRisk from './ReproducibilityRisk';
import AgenticPlanning from './AgenticPlanning';
import DnaHelix from './DnaHelix';
import { experimentsAPI, healthAPI } from '../services/api';
import { Database, Search, Upload, Beaker, Activity, Zap } from 'lucide-react';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('search');
  const [searchResults, setSearchResults] = useState(null);
  const [designResults, setDesignResults] = useState(null);
  const [agenticResults, setAgenticResults] = useState(null);
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    loadStats();
    checkHealth();
  }, []);

  const loadStats = async () => {
    try {
      const data = await experimentsAPI.getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const checkHealth = async () => {
    try {
      const data = await healthAPI.check();
      setHealth(data);
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  const handleSearchResults = (results) => {
    setSearchResults(results);
    setActiveTab('results');
  };

  const handleDesignResults = (results) => {
    setDesignResults(results);
    setActiveTab('design');
  };

  const handleUploadSuccess = () => {
    loadStats();
  };

  const handleAgenticResults = (results) => {
    setAgenticResults(results);
    setActiveTab('agentic-results');
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.logo}>
            <DnaHelix width={140} height={50} />
            <h1 style={styles.title}>BIOMEMORY</h1>
          </div>
          <div style={styles.healthStatus}>
            {health && (
              <div style={styles.healthBadge}>
                <Activity size={16} />
                <span>{health.status}</span>
                <span style={styles.healthDot(health.status)} />
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      {stats && (
        <div style={styles.statsBar}>
          <div style={styles.statCard}>
            <Database size={20} />
            <div>
              <div style={styles.statValue}>{stats.total_experiments}</div>
              <div style={styles.statLabel}>Total Experiments</div>
            </div>
          </div>
          <div style={styles.statCard}>
            <Activity size={20} />
            <div>
              <div style={styles.statValue}>{(stats.success_rate * 100).toFixed(0)}%</div>
              <div style={styles.statLabel}>Success Rate</div>
            </div>
          </div>
          <div style={styles.statCard}>
            <Beaker size={20} />
            <div>
              <div style={styles.statValue}>
                {Object.keys(stats.organism_distribution || {}).length}
              </div>
              <div style={styles.statLabel}>Organisms</div>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div style={styles.tabs}>
        <button
          style={activeTab === 'search' ? styles.tabActive : styles.tab}
          onClick={() => setActiveTab('search')}
        >
          <Search size={18} />
          Search
        </button>
        <button
          style={activeTab === 'agentic' ? styles.tabActive : styles.tab}
          onClick={() => setActiveTab('agentic')}
        >
          <Zap size={18} />
          AI Planning
        </button>
        <button
          style={activeTab === 'upload' ? styles.tabActive : styles.tab}
          onClick={() => setActiveTab('upload')}
        >
          <Upload size={18} />
          Upload
        </button>
        {searchResults && (
          <button
            style={activeTab === 'results' ? styles.tabActive : styles.tab}
            onClick={() => setActiveTab('results')}
          >
            <Database size={18} />
            Results ({searchResults.total_results})
          </button>
        )}
        {designResults && (
          <button
            style={activeTab === 'design' ? styles.tabActive : styles.tab}
            onClick={() => setActiveTab('design')}
          >
            <Beaker size={18} />
            Design ({designResults.variants?.length || 0})
          </button>
        )}
        {agenticResults && (
          <button
            style={activeTab === 'agentic-results' ? styles.tabActive : styles.tab}
            onClick={() => setActiveTab('agentic-results')}
          >
            <Zap size={18} />
            Planning Results
          </button>
        )}
      </div>

      {/* Main Content */}
      <div style={styles.content}>
        {activeTab === 'search' && (
          <SearchPanel
            onSearchResults={handleSearchResults}
            onDesignResults={handleDesignResults}
          />
        )}

        {activeTab === 'agentic' && (
          <AgenticPlanning onResults={handleAgenticResults} />
        )}

        {activeTab === 'upload' && (
          <ExperimentUpload onUploadSuccess={handleUploadSuccess} />
        )}

        {activeTab === 'results' && searchResults && (
          <div style={styles.resultsLayout}>
            <div style={styles.resultsMain}>
              <SimilarExperiments results={searchResults.results} />
            </div>
            <div style={styles.resultsSidebar}>
              <ReproducibilityRisk
                risk={searchResults.reproducibility_risk}
                metadata={searchResults.search_metadata}
              />
            </div>
          </div>
        )}

        {activeTab === 'design' && designResults && (
          <div style={styles.resultsLayout}>
            <div style={styles.resultsMain}>
              <DesignSuggestions variants={designResults.variants} />
            </div>
            <div style={styles.resultsSidebar}>
              <ReproducibilityRisk
                risk={designResults.reproducibility_risk}
                metadata={designResults.generation_metadata}
              />
            </div>
          </div>
        )}

        {activeTab === 'agentic-results' && agenticResults && (
          <div style={styles.resultsContainer}>
            <h2 style={styles.resultsTitle}>Agentic Planning Results</h2>

            {/* Status */}
            {agenticResults.status && (
              <div style={styles.agSection}>
                <span style={{
                  ...styles.agStatusBadge,
                  background: agenticResults.status === 'success' ? '#d1fae5' : '#fee2e2',
                  color: agenticResults.status === 'success' ? '#065f46' : '#991b1b',
                }}>
                  {agenticResults.status === 'success' ? 'Success' : agenticResults.status}
                </span>
              </div>
            )}

            {/* Parsed Experiment */}
            {agenticResults.parsed_experiment && (
              <div style={styles.agSection}>
                <h3 style={styles.agSectionTitle}>Parsed Experiment</h3>
                <div style={styles.agFieldGrid}>
                  {Object.entries(agenticResults.parsed_experiment).map(([key, value]) => (
                    value && (
                      <div key={key} style={styles.agField}>
                        <div style={styles.agFieldLabel}>{key.replace(/_/g, ' ')}</div>
                        <div style={styles.agFieldValue}>{String(value)}</div>
                      </div>
                    )
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {agenticResults.recommendations && (
              <div style={styles.agSection}>
                <h3 style={styles.agSectionTitle}>AI Recommendations</h3>
                <p style={styles.agRecommendation}>{agenticResults.recommendations}</p>
              </div>
            )}

            {/* Qdrant Insights */}
            {agenticResults.qdrant_insights && (
              <div style={styles.agSection}>
                <h3 style={styles.agSectionTitle}>Qdrant Insights</h3>
                <div style={styles.agFieldGrid}>
                  {Object.entries(agenticResults.qdrant_insights)
                    .filter(([, v]) => {
                      if (v == null) return false;
                      if (Array.isArray(v) && v.length === 0) return false;
                      if (typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === 0) return false;
                      return true;
                    })
                    .map(([key, value]) => (
                      <div key={key} style={styles.agField}>
                        <div style={styles.agFieldLabel}>{key.replace(/_/g, ' ')}</div>
                        <div style={styles.agFieldValue}>
                          {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #0a1131 0%, #0a1131 100%)',
  },
  header: {
    background: 'rgba(49, 48, 145, 0.1)',
    backdropFilter: 'blur(10px)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
    padding: '1rem 2rem',
  },
  headerContent: {
    maxWidth: '1400px',
    margin: '0 auto',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  title: {
    color: '#fff',
    fontSize: '1.8rem',
    fontWeight: '700',
  },
  healthStatus: {
    display: 'flex',
    gap: '1rem',
  },
  healthBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '0.5rem 1rem',
    background: 'rgba(38, 62, 129, 0.2)',
    borderRadius: '20px',
    color: '#fff',
    fontSize: '0.9rem',
  },
  healthDot: (status) => ({
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    background: status === 'healthy' ? '#0c682e' : status === 'degraded' ? '#dfb95a' : '#be0000',
  }),
  statsBar: {
    maxWidth: '1400px',
    margin: '2rem auto',
    padding: '0 2rem',
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '1rem',
  },
  statCard: {
    background: 'rgba(255, 255, 255, 0.95)',
    padding: '1.5rem',
    borderRadius: '12px',
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  },
  statValue: {
    fontSize: '2rem',
    fontWeight: '700',
    color: '#0a1131',
  },
  statLabel: {
    fontSize: '0.85rem',
    color: '#221f1f',
    textTransform: 'uppercase',
  },
  tabs: {
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '0 2rem',
    display: 'flex',
    gap: '0.5rem',
  },
  tab: {
    background: 'rgba(255, 255, 255, 0.2)',
    border: 'none',
    padding: '0.75rem 1.5rem',
    borderRadius: '8px 8px 0 0',
    color: '#fff',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    fontSize: '0.95rem',
    transition: 'all 0.2s',
  },
  tabActive: {
    background: '#fff',
    border: 'none',
    padding: '0.75rem 1.5rem',
    borderRadius: '8px 8px 0 0',
    color: '#0a1131',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    fontSize: '0.95rem',
    fontWeight: '600',
  },
  content: {
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '2rem',
    background: '#fff',
    minHeight: 'calc(100vh - 300px)',
    borderRadius: '12px 12px 0 0',
  },
  resultsLayout: {
    display: 'grid',
    gridTemplateColumns: '2fr 1fr',
    gap: '2rem',
  },
  resultsMain: {
    minHeight: '400px',
  },
  resultsSidebar: {
    minHeight: '400px',
  },
  resultsContainer: {
    minHeight: '400px',
  },
  resultsTitle: {
    fontSize: '1.5rem',
    fontWeight: '600',
    color: '#0a1131',
    marginBottom: '1rem',
  },
  resultsContent: {
    overflow: 'auto',
    maxHeight: '600px',
  },
  pre: {
    background: '#f5f5f5',
    padding: '1rem',
    borderRadius: '8px',
    fontSize: '0.85rem',
    color: '#333',
    overflow: 'auto',
    margin: 0,
  },
  agSection: {
    background: '#f9fafb',
    border: '1px solid #e5e7eb',
    borderRadius: '12px',
    padding: '1.5rem',
    marginBottom: '1.5rem',
    display: 'flex',
    flexWrap: 'wrap',
    gap: '0.75rem',
    alignItems: 'flex-start',
  },
  agSectionTitle: {
    width: '100%',
    fontSize: '1.15rem',
    fontWeight: '600',
    color: '#0a1131',
    marginBottom: '0.5rem',
  },
  agStatusBadge: {
    padding: '0.4rem 1rem',
    borderRadius: '20px',
    fontWeight: '600',
    fontSize: '0.95rem',
  },
  agPipelineTime: {
    padding: '0.4rem 1rem',
    borderRadius: '20px',
    background: '#f0fdf4',
    color: '#059669',
    fontWeight: '500',
    fontSize: '0.95rem',
  },
  agFieldGrid: {
    width: '100%',
    display: 'grid',
    gridTemplateColumns: '1fr',
    gap: '0.75rem',
  },
  agField: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
  },
  agFieldLabel: {
    fontSize: '0.85rem',
    fontWeight: '600',
    color: '#6b7280',
    textTransform: 'capitalize',
  },
  agFieldValue: {
    fontSize: '1.05rem',
    color: '#1f2937',
    lineHeight: '1.6',
    wordBreak: 'break-word',
    whiteSpace: 'pre-wrap',
  },
  agRecommendation: {
    width: '100%',
    fontSize: '1.05rem',
    color: '#1f2937',
    lineHeight: '1.7',
    wordBreak: 'break-word',
    whiteSpace: 'pre-wrap',
    margin: 0,
  },
};
