/**
 * ExperimentUpload Component
 * Upload new biological experiments
 */
import { useState } from 'react';
import { experimentsAPI } from '../services/api';
import { Upload, CheckCircle, XCircle } from 'lucide-react';

export default function ExperimentUpload({ onUploadSuccess }) {
  const [formData, setFormData] = useState({
    text: '',
    sequence: '',
    organism: '',
    temperature: '',
    ph: '',
    success: true,
    notes: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedId, setUploadedId] = useState(null);

  const formatValidationErrors = (errors) => {
    if (Array.isArray(errors)) {
      return errors.map(error => {
        const field = error.loc?.[error.loc.length - 1] || 'field';
        return `${field}: ${error.msg}`;
      }).join('\n');
    }
    return errors;
  };

  const handleChange = (e) => {
    const value = e.target.type === 'checkbox' ? e.target.checked : e.target.value;
    setFormData({
      ...formData,
      [e.target.name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setUploadedId(null);

    try {
      const experimentData = {
        text: formData.text,
        sequence: formData.sequence || undefined,
        conditions: {},
        success: formData.success,
        notes: formData.notes || undefined,
      };

      if (formData.organism) experimentData.conditions.organism = formData.organism;
      if (formData.temperature) experimentData.conditions.temperature = parseFloat(formData.temperature);
      if (formData.ph) experimentData.conditions.ph = parseFloat(formData.ph);

      if (Object.keys(experimentData.conditions).length === 0) {
        delete experimentData.conditions;
      }

      const result = await experimentsAPI.upload(experimentData);
      setUploadedId(result.experiment_id);

      // Reset form
      setFormData({
        text: '',
        sequence: '',
        organism: '',
        temperature: '',
        ph: '',
        success: true,
        notes: '',
      });

      onUploadSuccess();
    } catch (err) {
      const errorDetail = err.response?.data?.detail;
      setError(formatValidationErrors(errorDetail) || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>
        <Upload size={24} />
        Upload New Experiment
      </h2>

      {uploadedId && (
        <div style={styles.success}>
          <CheckCircle size={20} />
          Experiment uploaded successfully! ID: {uploadedId}
        </div>
      )}

      <form onSubmit={handleSubmit} style={styles.form}>
        {/* Text Description */}
        <div style={styles.field}>
          <label style={styles.label}>Experiment Description *</label>
          <textarea
            name="text"
            value={formData.text}
            onChange={handleChange}
            placeholder="Describe your experiment in detail..."
            style={styles.textarea}
            rows={5}
            required
          />
        </div>

        {/* Sequence */}
        <div style={styles.field}>
          <label style={styles.label}>Biological Sequence (optional)</label>
          <input
            type="text"
            name="sequence"
            value={formData.sequence}
            onChange={handleChange}
            placeholder="ATGGCTAGCAAAGGAGAAG..."
            style={styles.input}
          />
        </div>

        {/* Conditions */}
        <div style={styles.conditionsGrid}>
          <div style={styles.field}>
            <label style={styles.label}>Organism</label>
            <input
              type="text"
              name="organism"
              value={formData.organism}
              onChange={handleChange}
              placeholder="e.g., human, ecoli, mouse"
              style={styles.input}
            />
          </div>
          <div style={styles.field}>
            <label style={styles.label}>Temperature (°C)</label>
            <input
              type="number"
              name="temperature"
              value={formData.temperature}
              onChange={handleChange}
              placeholder="37"
              style={styles.input}
              step="0.1"
            />
          </div>
          <div style={styles.field}>
            <label style={styles.label}>pH</label>
            <input
              type="number"
              name="ph"
              value={formData.ph}
              onChange={handleChange}
              placeholder="7.0"
              style={styles.input}
              step="0.1"
              min="0"
              max="14"
            />
          </div>
        </div>

        {/* Success Checkbox */}
        <div style={styles.checkboxField}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              name="success"
              checked={formData.success}
              onChange={handleChange}
              style={styles.checkbox}
            />
            <span style={formData.success ? styles.successBadge : styles.failureBadge}>
              {formData.success ? (
                <>
                  <CheckCircle size={16} /> Successful Experiment
                </>
              ) : (
                <>
                  <XCircle size={16} /> Failed Experiment
                </>
              )}
            </span>
          </label>
        </div>

        {/* Notes */}
        <div style={styles.field}>
          <label style={styles.label}>Additional Notes (optional)</label>
          <textarea
            name="notes"
            value={formData.notes}
            onChange={handleChange}
            placeholder="Add any additional observations, troubleshooting notes, or comments..."
            style={styles.textarea}
            rows={3}
          />
        </div>

        {/* Error Display */}
        {error && <div style={styles.error}>{error}</div>}

        {/* Submit Button */}
        <button type="submit" disabled={loading} style={styles.submitButton}>
          {loading ? 'Uploading...' : (
            <>
              <Upload size={20} />
              Upload Experiment
            </>
          )}
        </button>
      </form>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '900px',
    margin: '0 auto',
  },
  title: {
    fontSize: '1.8rem',
    color: '#333',
    marginBottom: '1.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  success: {
    padding: '1rem',
    background: '#d1fae5',
    border: '1px solid #6ee7b7',
    borderRadius: '8px',
    color: '#065f46',
    marginBottom: '1.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  form: {
    background: '#f9fafb',
    padding: '2rem',
    borderRadius: '12px',
    border: '1px solid #e5e7eb',
  },
  field: {
    marginBottom: '1.5rem',
  },
  label: {
    display: 'block',
    marginBottom: '0.5rem',
    color: '#374151',
    fontWeight: '600',
    fontSize: '0.95rem',
  },
  input: {
    width: '100%',
    padding: '0.75rem',
    border: '2px solid #d1d5db',
    borderRadius: '8px',
    fontSize: '1rem',
  },
  textarea: {
    width: '100%',
    padding: '0.75rem',
    border: '2px solid #d1d5db',
    borderRadius: '8px',
    fontSize: '1rem',
    resize: 'vertical',
    fontFamily: 'inherit',
  },
  conditionsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '1rem',
    marginBottom: '1.5rem',
  },
  checkboxField: {
    marginBottom: '1.5rem',
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    cursor: 'pointer',
  },
  checkbox: {
    width: '20px',
    height: '20px',
    cursor: 'pointer',
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
  },
  error: {
    padding: '1rem',
    background: '#fee2e2',
    border: '1px solid #fecaca',
    borderRadius: '8px',
    color: '#991b1b',
    marginBottom: '1rem',
    whiteSpace: 'pre-line',
  },
  submitButton: {
    width: '100%',
    padding: '1rem',
    background: 'linear-gradient(135deg, #0a1131 0%, #0a1131 100%)',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1.1rem',
    fontWeight: '600',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.5rem',
  },
};
