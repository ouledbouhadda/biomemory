/**
 * ReproducibilityRisk Component
 * Display reproducibility risk analysis
 */
import { AlertTriangle, CheckCircle, Info } from 'lucide-react';

export default function ReproducibilityRisk({ risk, metadata }) {
  const getRiskLevel = (riskScore) => {
    if (riskScore < 0.3) return 'low';
    if (riskScore < 0.6) return 'medium';
    return 'high';
  };

  const getRiskColor = (level) => {
    const colors = {
      low: { bg: '#d1fae5', color: '#065f46', icon: <CheckCircle size={24} /> },
      medium: { bg: '#fef3c7', color: '#92400e', icon: <AlertTriangle size={24} /> },
      high: { bg: '#fee2e2', color: '#991b1b', icon: <AlertTriangle size={24} /> },
    };
    return colors[level] || colors.medium;
  };

  const riskLevel = getRiskLevel(risk);
  const riskStyle = getRiskColor(riskLevel);
  const riskPercentage = (risk * 100).toFixed(1);

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>
        <Info size={20} />
        Reproducibility Analysis
      </h3>

      {/* Risk Score */}
      <div style={{ ...styles.riskCard, background: riskStyle.bg, color: riskStyle.color }}>
        <div style={styles.riskIcon}>{riskStyle.icon}</div>
        <div style={styles.riskContent}>
          <div style={styles.riskLabel}>Reproducibility Risk</div>
          <div style={styles.riskScore}>{riskPercentage}%</div>
          <div style={styles.riskLevel}>{riskLevel.toUpperCase()}</div>
        </div>
      </div>

      {/* Risk Gauge */}
      <div style={styles.gauge}>
        <div style={styles.gaugeTrack}>
          <div style={{ ...styles.gaugeFill, width: `${riskPercentage}%`, background: riskStyle.color }} />
        </div>
        <div style={styles.gaugeLabels}>
          <span>Low</span>
          <span>Medium</span>
          <span>High</span>
        </div>
      </div>

      {/* Interpretation */}
      <div style={styles.interpretation}>
        <h4 style={styles.interpretationTitle}>What does this mean?</h4>
        {riskLevel === 'low' && (
          <p style={styles.interpretationText}>
            Similar experiments have shown <strong>consistent results</strong>. High confidence in
            reproducibility.
          </p>
        )}
        {riskLevel === 'medium' && (
          <p style={styles.interpretationText}>
            Similar experiments show <strong>moderate variability</strong>. Pay attention to
            experimental conditions.
          </p>
        )}
        {riskLevel === 'high' && (
          <p style={styles.interpretationText}>
            Similar experiments have <strong>inconsistent outcomes</strong>. Consider reviewing
            protocol details carefully.
          </p>
        )}
      </div>

      {/* Metadata */}
      {metadata && (
        <div style={styles.metadata}>
          <h4 style={styles.metadataTitle}>Analysis Details</h4>
          <div style={styles.metadataGrid}>
            {metadata.modalities_used && (
              <div style={styles.metadataItem}>
                <div style={styles.metadataLabel}>Modalities Used:</div>
                <div style={styles.metadataValue}>
                  {Object.entries(metadata.modalities_used)
                    .filter(([, value]) => value)
                    .map(([key]) => key.replace('has_', ''))
                    .join(', ') || 'none'}
                </div>
              </div>
            )}
            {metadata.search_strategy && (
              <div style={styles.metadataItem}>
                <div style={styles.metadataLabel}>Strategy:</div>
                <div style={styles.metadataValue}>{metadata.search_strategy}</div>
              </div>
            )}
            {metadata.method && (
              <div style={styles.metadataItem}>
                <div style={styles.metadataLabel}>Method:</div>
                <div style={styles.metadataValue}>{metadata.method}</div>
              </div>
            )}
            {metadata.model && (
              <div style={styles.metadataItem}>
                <div style={styles.metadataLabel}>Model:</div>
                <div style={styles.metadataValue}>{metadata.model}</div>
              </div>
            )}
            {metadata.risk_level && (
              <div style={styles.metadataItem}>
                <div style={styles.metadataLabel}>Risk Level:</div>
                <div style={styles.metadataValue}>{metadata.risk_level}</div>
              </div>
            )}
            {metadata.total_analyzed !== undefined && (
              <div style={styles.metadataItem}>
                <div style={styles.metadataLabel}>Experiments Analyzed:</div>
                <div style={styles.metadataValue}>{metadata.total_analyzed}</div>
              </div>
            )}
            {metadata.context_experiments !== undefined && (
              <div style={styles.metadataItem}>
                <div style={styles.metadataLabel}>Context Size:</div>
                <div style={styles.metadataValue}>{metadata.context_experiments} experiments</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div style={styles.recommendations}>
        <h4 style={styles.recommendationsTitle}>Recommendations</h4>
        <ul style={styles.recommendationsList}>
          {metadata?.recommendations && metadata.recommendations.length > 0 ? (
            metadata.recommendations.map((rec, i) => (
              <li key={i}>{riskLevel === 'low' ? '✓' : '!'} {rec}</li>
            ))
          ) : (
            <>
              {riskLevel === 'low' && (
                <>
                  <li>✓ Proceed with confidence</li>
                  <li>✓ Standard protocol controls should suffice</li>
                </>
              )}
              {riskLevel === 'medium' && (
                <>
                  <li>! Include additional controls</li>
                  <li>! Document conditions precisely</li>
                  <li>! Consider replicates</li>
                </>
              )}
              {riskLevel === 'high' && (
                <>
                  <li>! Review similar failed experiments</li>
                  <li>! Validate critical parameters</li>
                  <li>! Consider pilot experiments</li>
                </>
              )}
            </>
          )}
        </ul>
      </div>
    </div>
  );
}

const styles = {
  container: {
    background: '#fff',
    border: '1px solid #e5e7eb',
    borderRadius: '12px',
    padding: '1.5rem',
  },
  title: {
    fontSize: '1.2rem',
    fontWeight: '600',
    color: '#374151',
    marginBottom: '1.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  riskCard: {
    padding: '1.5rem',
    borderRadius: '12px',
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    marginBottom: '1.5rem',
  },
  riskIcon: {
    flexShrink: 0,
  },
  riskContent: {
    flex: 1,
  },
  riskLabel: {
    fontSize: '0.85rem',
    fontWeight: '600',
    textTransform: 'uppercase',
    opacity: 0.8,
  },
  riskScore: {
    fontSize: '2.5rem',
    fontWeight: '700',
    lineHeight: 1,
    marginTop: '0.25rem',
  },
  riskLevel: {
    fontSize: '0.9rem',
    fontWeight: '600',
    marginTop: '0.25rem',
  },
  gauge: {
    marginBottom: '1.5rem',
  },
  gaugeTrack: {
    height: '12px',
    background: '#e5e7eb',
    borderRadius: '6px',
    overflow: 'hidden',
    marginBottom: '0.5rem',
  },
  gaugeFill: {
    height: '100%',
    transition: 'width 0.5s ease',
    borderRadius: '6px',
  },
  gaugeLabels: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.75rem',
    color: '#6b7280',
  },
  interpretation: {
    padding: '1rem',
    background: '#f9fafb',
    borderRadius: '8px',
    marginBottom: '1.5rem',
  },
  interpretationTitle: {
    fontSize: '0.95rem',
    fontWeight: '600',
    color: '#374151',
    marginBottom: '0.5rem',
  },
  interpretationText: {
    fontSize: '0.9rem',
    color: '#6b7280',
    lineHeight: '1.6',
    margin: 0,
  },
  metadata: {
    marginBottom: '1.5rem',
  },
  metadataTitle: {
    fontSize: '0.95rem',
    fontWeight: '600',
    color: '#374151',
    marginBottom: '0.75rem',
  },
  metadataGrid: {
    display: 'grid',
    gap: '0.75rem',
  },
  metadataItem: {
    fontSize: '0.85rem',
  },
  metadataLabel: {
    color: '#6b7280',
    fontWeight: '600',
    marginBottom: '0.25rem',
  },
  metadataValue: {
    color: '#374151',
  },
  recommendations: {
    padding: '1rem',
    background: '#eff6ff',
    borderRadius: '8px',
  },
  recommendationsTitle: {
    fontSize: '0.95rem',
    fontWeight: '600',
    color: '#0a1131',
    marginBottom: '0.75rem',
  },
  recommendationsList: {
    margin: 0,
    paddingLeft: '1.5rem',
    color: '#0a1131',
    fontSize: '0.9rem',
  },
};
