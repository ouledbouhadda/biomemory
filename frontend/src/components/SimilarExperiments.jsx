/**
 * SimilarExperiments Component
 * Display search results
 */
import { CheckCircle, XCircle, Database, ExternalLink } from 'lucide-react';

export default function SimilarExperiments({ results }) {
  if (!results || results.length === 0) {
    return (
      <div style={styles.empty}>
        <Database size={48} color="#ccc" />
        <p>No similar experiments found</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>
        <Database size={24} />
        Similar Experiments ({results.length})
      </h2>

      <div style={styles.results}>
        {results.map((exp, index) => (
          <div key={exp.experiment_id} style={styles.card}>
            {/* Header */}
            <div style={styles.cardHeader}>
              <div style={styles.rank}>#{index + 1}</div>
              <div style={styles.scores}>
                <div style={styles.score}>
                  Similarity: <strong>{(exp.similarity_score * 100).toFixed(1)}%</strong>
                </div>
                {exp.reranked_score && (
                  <div style={styles.score}>
                    Re-ranked: <strong>{(exp.reranked_score * 100).toFixed(1)}%</strong>
                  </div>
                )}
              </div>
              <div style={exp.success ? styles.successBadge : styles.failureBadge}>
                {exp.success ? (
                  <>
                    <CheckCircle size={16} /> Success
                  </>
                ) : (
                  <>
                    <XCircle size={16} /> Failed
                  </>
                )}
              </div>
            </div>

            {/* Content */}
            <div style={styles.cardContent}>
              <p style={styles.text}>{exp.text}</p>

              {exp.sequence && (
                <div style={styles.sequence}>
                  <strong>Sequence:</strong> {exp.sequence.substring(0, 60)}
                  {exp.sequence.length > 60 && '...'}
                </div>
              )}

              {exp.conditions && (
                <div style={styles.conditions}>
                  {exp.conditions.organism && (
                    <span style={styles.conditionTag}>
                      🦠 {exp.conditions.organism}
                    </span>
                  )}
                  {exp.conditions.temperature && (
                    <span style={styles.conditionTag}>
                      🌡️ {exp.conditions.temperature}°C
                    </span>
                  )}
                  {exp.conditions.ph && (
                    <span style={styles.conditionTag}>⚗️ pH {exp.conditions.ph}</span>
                  )}
                </div>
              )}
            </div>

            {/* Footer with Evidence */}
            <div style={styles.cardFooter}>
              <div style={styles.source}>
                <strong>Source:</strong> {exp.source_type || 'unknown'}
              </div>
              <div style={styles.verification}>
                {getVerificationBadge(exp.verification_status)}
              </div>
              {exp.publication && (
                <div style={styles.publication}>
                  <ExternalLink size={14} />
                  {exp.publication}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function getVerificationBadge(status) {
  const badges = {
    peer_reviewed: {
      text: '✓ Peer Reviewed',
      style: { background: '#d1fae5', color: '#065f46' },
    },
    user_generated: {
      text: '👤 User Generated',
      style: { background: '#dbeafe', color: '#1e40af' },
    },
    unverified: {
      text: '⚠ Unverified',
      style: { background: '#fef3c7', color: '#92400e' },
    },
  };

  const badge = badges[status] || badges.unverified;

  return (
    <span style={{ ...styles.verificationBadge, ...badge.style }}>{badge.text}</span>
  );
}

const styles = {
  container: {},
  title: {
    fontSize: '1.5rem',
    color: '#333',
    marginBottom: '1.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  empty: {
    textAlign: 'center',
    padding: '4rem 2rem',
    color: '#999',
  },
  results: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  card: {
    background: '#fff',
    border: '1px solid #e5e7eb',
    borderRadius: '12px',
    padding: '1.5rem',
    transition: 'box-shadow 0.2s',
    cursor: 'pointer',
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    marginBottom: '1rem',
    paddingBottom: '1rem',
    borderBottom: '1px solid #e5e7eb',
  },
  rank: {
    fontSize: '1.5rem',
    fontWeight: '700',
    color: '#0a1131',
    minWidth: '50px',
  },
  scores: {
    flex: 1,
    display: 'flex',
    gap: '1rem',
  },
  score: {
    fontSize: '0.9rem',
    color: '#6b7280',
  },
  successBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '0.5rem 1rem',
    background: '#d1fae5',
    color: '#065f46',
    borderRadius: '20px',
    fontWeight: '600',
    fontSize: '0.9rem',
  },
  failureBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '0.5rem 1rem',
    background: '#fee2e2',
    color: '#991b1b',
    borderRadius: '20px',
    fontWeight: '600',
    fontSize: '0.9rem',
  },
  cardContent: {
    marginBottom: '1rem',
  },
  text: {
    fontSize: '1rem',
    color: '#374151',
    lineHeight: '1.6',
    marginBottom: '1rem',
  },
  sequence: {
    padding: '0.75rem',
    background: '#f3f4f6',
    borderRadius: '6px',
    fontSize: '0.85rem',
    fontFamily: 'monospace',
    marginBottom: '0.75rem',
    color: '#1f2937',
  },
  conditions: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '0.5rem',
  },
  conditionTag: {
    padding: '0.25rem 0.75rem',
    background: '#eff6ff',
    color: '#0a1131',
    borderRadius: '12px',
    fontSize: '0.85rem',
  },
  cardFooter: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    paddingTop: '1rem',
    borderTop: '1px solid #e5e7eb',
    fontSize: '0.85rem',
    color: '#6b7280',
  },
  source: {},
  verification: {},
  verificationBadge: {
    padding: '0.25rem 0.75rem',
    borderRadius: '12px',
    fontSize: '0.8rem',
    fontWeight: '600',
  },
  publication: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem',
    color: '#0a1131',
    textDecoration: 'none',
  },
};
