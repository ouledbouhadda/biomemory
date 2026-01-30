/**
 * SimilarExperiments Component
 * Enriched search results display with feedback
 */
import { useState } from 'react';
import { CheckCircle, XCircle, Database, ExternalLink, ThumbsUp, ThumbsDown, Beaker, FlaskConical, User } from 'lucide-react';

export default function SimilarExperiments({ results, searchQuery }) {
  const [feedbackGiven, setFeedbackGiven] = useState({});

  if (!results || results.length === 0) {
    return (
      <div style={styles.empty}>
        <Database size={48} color="#ccc" />
        <p>No similar experiments found</p>
      </div>
    );
  }

  const handleFeedback = async (experimentId, feedback) => {
    try {
      const response = await fetch('/api/v1/search/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token') || ''}`
        },
        body: JSON.stringify({
          experiment_id: experimentId,
          feedback: feedback,
          query_text: searchQuery || ''
        })
      });
      if (response.ok) {
        setFeedbackGiven(prev => ({ ...prev, [experimentId]: feedback }));
      }
    } catch (err) {
      console.error('Feedback failed:', err);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>
        <Database size={24} />
        Similar Experiments ({results.length})
      </h2>

      <div style={styles.results}>
        {results.map((exp, index) => (
          <div key={exp.experiment_id || index} style={styles.card}>
            {/* Header */}
            <div style={styles.cardHeader}>
              <div style={styles.rank}>#{index + 1}</div>
              <div style={styles.scores}>
                <div style={styles.score}>
                  Similarity: <strong>{(exp.similarity_score * 100).toFixed(1)}%</strong>
                </div>
                {exp.reranked_score != null && (
                  <div style={styles.score}>
                    Re-ranked: <strong>{(exp.reranked_score * 100).toFixed(1)}%</strong>
                  </div>
                )}
              </div>
              {exp.success === true ? (
                <div style={styles.successBadge}>
                  <CheckCircle size={16} /> Success
                </div>
              ) : exp.success === false ? (
                <div style={styles.failureBadge}>
                  <XCircle size={16} /> Failed
                </div>
              ) : (
                <div style={styles.unknownBadge}>
                  <Database size={16} /> Experiment
                </div>
              )}
            </div>

            {/* Content */}
            <div style={styles.cardContent}>
              {exp.title && (
                <h3 style={styles.expTitle}>{exp.title}</h3>
              )}
              <p style={styles.text}>
                {exp.text || exp.content || 'No description available'}
              </p>

              {exp.sequence && (
                <div style={styles.sequence}>
                  <strong>Sequence:</strong> {exp.sequence.substring(0, 60)}
                  {exp.sequence.length > 60 && '...'}
                </div>
              )}

              {/* Conditions tags */}
              <div style={styles.conditions}>
                {(exp.conditions?.organism || exp.organism) && (
                  <span style={styles.conditionTag}>
                    {exp.conditions?.organism || exp.organism}
                  </span>
                )}
                {(exp.conditions?.temperature || exp.temperature) && (
                  <span style={styles.conditionTag}>
                    {exp.conditions?.temperature || exp.temperature}
                  </span>
                )}
                {(exp.conditions?.ph || exp.ph) && (
                  <span style={styles.conditionTag}>
                    pH {exp.conditions?.ph || exp.ph}
                  </span>
                )}
                {(exp.assay) && (
                  <span style={{...styles.conditionTag, ...styles.assayTag}}>
                    <FlaskConical size={12} /> {exp.assay}
                  </span>
                )}
                {exp.type && (
                  <span style={styles.conditionTag}>
                    {exp.type}
                  </span>
                )}
              </div>
            </div>

            {/* Footer with Evidence + Feedback */}
            <div style={styles.cardFooter}>
              <div style={styles.footerLeft}>
                <div style={styles.source}>
                  <strong>Source:</strong> {exp.source || exp.source_type || 'unknown'}
                </div>
                <div style={styles.verification}>
                  {getVerificationBadge(exp.verification_status)}
                </div>
                {exp.reference && (
                  <div style={styles.publication}>
                    <ExternalLink size={14} />
                    <a
                      href={exp.reference}
                      style={styles.refText}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {exp.reference}
                    </a>
                  </div>
                )}
                {exp.contact && (
                  <div style={styles.contactInfo}>
                    <User size={12} /> {exp.contact}
                  </div>
                )}
              </div>

              {/* Feedback buttons */}
              <div style={styles.feedbackButtons}>
                {feedbackGiven[exp.experiment_id] ? (
                  <span style={styles.feedbackDone}>
                    {feedbackGiven[exp.experiment_id] === 'like' ? 'Relevant' : 'Not relevant'}
                  </span>
                ) : (
                  <>
                    <button
                      style={styles.feedbackBtn}
                      onClick={() => handleFeedback(exp.experiment_id, 'like')}
                      title="Relevant result"
                    >
                      <ThumbsUp size={14} />
                    </button>
                    <button
                      style={styles.feedbackBtnNeg}
                      onClick={() => handleFeedback(exp.experiment_id, 'dislike')}
                      title="Not relevant"
                    >
                      <ThumbsDown size={14} />
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function getVerificationBadge(status) {
  const badges = {
    peer_reviewed: { text: 'Peer Reviewed', style: { background: '#d1fae5', color: '#065f46' } },
    user_generated: { text: 'User Generated', style: { background: '#dbeafe', color: '#1e40af' } },
  };
    if (!status || !badges[status]) {
      return null;
    }
  const badge = badges[status];
  return (
    <span style={{ ...styles.verificationBadge, ...badge.style }}>{badge.text}</span>
  );
}

const styles = {
  container: {},
  title: { fontSize: '1.5rem', color: '#333', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' },
  empty: { textAlign: 'center', padding: '4rem 2rem', color: '#999' },
  results: { display: 'flex', flexDirection: 'column', gap: '1rem' },
  card: { background: '#fff', border: '1px solid #e5e7eb', borderRadius: '12px', padding: '1.5rem', transition: 'box-shadow 0.2s' },
  cardHeader: { display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem', paddingBottom: '1rem', borderBottom: '1px solid #e5e7eb' },
  rank: { fontSize: '1.5rem', fontWeight: '700', color: '#0a1131', minWidth: '50px' },
  scores: { flex: 1, display: 'flex', gap: '1rem' },
  score: { fontSize: '0.9rem', color: '#6b7280' },
  successBadge: { display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem', background: '#d1fae5', color: '#065f46', borderRadius: '20px', fontWeight: '600', fontSize: '0.9rem' },
  failureBadge: { display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem', background: '#fee2e2', color: '#991b1b', borderRadius: '20px', fontWeight: '600', fontSize: '0.9rem' },
  unknownBadge: { display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem', background: '#f3f4f6', color: '#4b5563', borderRadius: '20px', fontWeight: '600', fontSize: '0.9rem' },
  cardContent: { marginBottom: '1rem' },
  expTitle: { fontSize: '1.1rem', fontWeight: '600', color: '#1f2937', marginBottom: '0.5rem' },
  text: { fontSize: '1rem', color: '#374151', lineHeight: '1.6', marginBottom: '1rem' },
  sequence: { padding: '0.75rem', background: '#f3f4f6', borderRadius: '6px', fontSize: '0.85rem', fontFamily: 'monospace', marginBottom: '0.75rem', color: '#1f2937' },
  conditions: { display: 'flex', flexWrap: 'wrap', gap: '0.5rem' },
  conditionTag: { padding: '0.25rem 0.75rem', background: '#eff6ff', color: '#0a1131', borderRadius: '12px', fontSize: '0.85rem', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' },
  assayTag: { background: '#f0fdf4', color: '#166534', border: '1px solid #bbf7d0' },
  cardFooter: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '1rem', borderTop: '1px solid #e5e7eb', fontSize: '0.85rem', color: '#6b7280' },
  footerLeft: { display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' },
  source: {},
  verification: {},
  verificationBadge: { padding: '0.25rem 0.75rem', borderRadius: '12px', fontSize: '0.8rem', fontWeight: '600' },
  publication: { display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#0a1131' },
  refText: { fontSize: '0.8rem', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  contactInfo: { display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.8rem', color: '#6b7280' },
  feedbackButtons: { display: 'flex', gap: '0.5rem', alignItems: 'center' },
  feedbackBtn: { padding: '0.4rem 0.6rem', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.3rem', color: '#166534', fontSize: '0.8rem' },
  feedbackBtnNeg: { padding: '0.4rem 0.6rem', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.3rem', color: '#991b1b', fontSize: '0.8rem' },
  feedbackDone: { fontSize: '0.8rem', color: '#059669', fontWeight: '500' },
};
