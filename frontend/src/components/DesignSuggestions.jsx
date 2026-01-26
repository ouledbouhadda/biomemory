/**
 * DesignSuggestions Component
 * Display AI-generated design variants
 */
import { Beaker, Lightbulb, AlertTriangle } from 'lucide-react';

export default function DesignSuggestions({ variants }) {
  if (!variants || variants.length === 0) {
    return (
      <div style={styles.empty}>
        <Beaker size={48} color="#ccc" />
        <p>No design variants generated</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>
        <Beaker size={24} />
        Design Variants ({variants.length})
      </h2>

      <div style={styles.variants}>
        {variants.map((variant, index) => (
          <div key={variant.variant_id || index} style={styles.card}>
            {/* Header */}
            <div style={styles.cardHeader}>
              <div style={styles.variantNumber}>Variant {index + 1}</div>
              <div style={styles.confidence}>
                Confidence: <strong>{(variant.confidence * 100).toFixed(1)}%</strong>
              </div>
            </div>

            {/* Description */}
            <div style={styles.cardContent}>
              <p style={styles.text}>{variant.text}</p>

              {variant.sequence && (
                <div style={styles.sequence}>
                  <strong>Modified Sequence:</strong> {variant.sequence.substring(0, 60)}
                  {variant.sequence.length > 60 && '...'}
                </div>
              )}

              {variant.conditions && (
                <div style={styles.conditions}>
                  {variant.conditions.organism && (
                    <span style={styles.conditionTag}>
                      🦠 {variant.conditions.organism}
                    </span>
                  )}
                  {variant.conditions.temperature && (
                    <span style={styles.conditionTag}>
                      🌡️ {variant.conditions.temperature}°C
                    </span>
                  )}
                  {variant.conditions.ph && (
                    <span style={styles.conditionTag}>
                      ⚗️ pH {variant.conditions.ph}
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Modifications */}
            {variant.modifications && variant.modifications.length > 0 && (
              <div style={styles.modifications}>
                <h4 style={styles.sectionTitle}>
                  <Lightbulb size={16} />
                  Key Modifications:
                </h4>
                <ul style={styles.modificationList}>
                  {variant.modifications.map((mod, idx) => (
                    <li key={idx} style={styles.modificationItem}>
                      {mod}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Justification */}
            {variant.justification && (
              <div style={styles.justification}>
                <h4 style={styles.sectionTitle}>
                  <Lightbulb size={16} />
                  Scientific Justification:
                </h4>
                <p style={styles.justificationText}>{variant.justification}</p>
              </div>
            )}

            {/* Risk Factors */}
            {variant.risk_factors && variant.risk_factors.length > 0 && (
              <div style={styles.riskFactors}>
                <h4 style={styles.riskTitle}>
                  <AlertTriangle size={16} />
                  Risk Factors:
                </h4>
                <ul style={styles.riskList}>
                  {variant.risk_factors.map((risk, idx) => (
                    <li key={idx} style={styles.riskItem}>
                      {risk}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Supporting Evidence */}
            {variant.supporting_evidence && variant.supporting_evidence.length > 0 && (
              <div style={styles.evidence}>
                <h4 style={styles.sectionTitle}>
                  Supporting Evidence ({variant.supporting_evidence.length} similar successes)
                </h4>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
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
  variants: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  card: {
    background: '#fff',
    border: '2px solid #0a1131',
    borderRadius: '12px',
    padding: '1.5rem',
  },
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1rem',
    paddingBottom: '1rem',
    borderBottom: '2px solid #eff6ff',
  },
  variantNumber: {
    fontSize: '1.3rem',
    fontWeight: '700',
    color: '#0a1131',
  },
  confidence: {
    fontSize: '0.95rem',
    color: '#6b7280',
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
  modifications: {
    marginBottom: '1rem',
    padding: '1rem',
    background: '#fef3c7',
    borderRadius: '8px',
  },
  sectionTitle: {
    fontSize: '0.95rem',
    fontWeight: '600',
    color: '#374151',
    marginBottom: '0.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  modificationList: {
    margin: 0,
    paddingLeft: '1.5rem',
  },
  modificationItem: {
    fontSize: '0.9rem',
    color: '#92400e',
    marginBottom: '0.25rem',
  },
  justification: {
    marginBottom: '1rem',
    padding: '1rem',
    background: '#eff6ff',
    borderRadius: '8px',
  },
  justificationText: {
    fontSize: '0.9rem',
    color: '#0a1131',
    lineHeight: '1.6',
    margin: 0,
  },
  riskFactors: {
    marginBottom: '1rem',
    padding: '1rem',
    background: '#fee2e2',
    borderRadius: '8px',
  },
  riskTitle: {
    fontSize: '0.95rem',
    fontWeight: '600',
    color: '#991b1b',
    marginBottom: '0.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  riskList: {
    margin: 0,
    paddingLeft: '1.5rem',
  },
  riskItem: {
    fontSize: '0.9rem',
    color: '#991b1b',
    marginBottom: '0.25rem',
  },
  evidence: {
    padding: '0.75rem',
    background: '#d1fae5',
    borderRadius: '8px',
  },
};
