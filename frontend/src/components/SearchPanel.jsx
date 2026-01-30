/**
 * SearchPanel Component
 * Multimodal search interface with advanced filters
 */
import { useState, useEffect } from 'react';
import { searchAPI, designAPI } from '../services/api';
import { Search, Dna, Image, Sliders, Beaker, Filter, ChevronDown, ChevronUp } from 'lucide-react';

export default function SearchPanel({ onSearchResults, onDesignResults }) {
  const [mode, setMode] = useState('search');
  const [formData, setFormData] = useState({
    text: '',
    sequence: '',
    image: null,
    organism: '',
    temperature: '',
    ph: '',
    numVariants: 3,
    // Advanced filters
    assay: '',
    source: '',
    type: '',
    successOnly: false,
    minSuccessRate: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchMode, setSearchMode] = useState('multimodal');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [facets, setFacets] = useState(null);

  // Load facets on mount
  useEffect(() => {
    loadFacets();
  }, []);

  const loadFacets = async () => {
    try {
      const token = localStorage.getItem('access_token') || '';
      const headers = { 'Authorization': `Bearer ${token}` };

      const [organismRes, assayRes, sourceRes] = await Promise.allSettled([
        fetch('/api/v1/search/facets/organism', { headers }).then(r => r.ok ? r.json() : null),
        fetch('/api/v1/search/facets/assay', { headers }).then(r => r.ok ? r.json() : null),
        fetch('/api/v1/search/facets/source', { headers }).then(r => r.ok ? r.json() : null),
      ]);

      setFacets({
        organisms: organismRes.status === 'fulfilled' ? organismRes.value : null,
        assays: assayRes.status === 'fulfilled' ? assayRes.value : null,
        sources: sourceRes.status === 'fulfilled' ? sourceRes.value : null,
      });
    } catch (err) {
      console.warn('Facets loading failed:', err);
    }
  };

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
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? checked : value,
    });
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setFormData({ ...formData, image: event.target.result });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleImageSearch = async (e) => {
    e.preventDefault();
    if (!formData.image) {
      setError('Please select an image first');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const results = await searchAPI.searchByImage(formData.image, {
        limit: 10,
        include_failures: true,
      });
      onSearchResults(results);
    } catch (err) {
      const errorDetail = err.response?.data?.detail;
      setError(formatValidationErrors(errorDetail) || 'Image search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const conditionsObj = {};
      if (formData.organism) conditionsObj.organism = formData.organism;
      if (formData.temperature) conditionsObj.temperature = parseFloat(formData.temperature);
      if (formData.ph) conditionsObj.ph = parseFloat(formData.ph);
      // Advanced filters
      if (formData.assay) conditionsObj.assay = formData.assay;
      if (formData.source) conditionsObj.source = formData.source;
      if (formData.type) conditionsObj.type = formData.type;
      if (formData.successOnly) conditionsObj.success_only = true;

      const searchPayload = {
        text: formData.text || '',
        sequence: formData.sequence || null,
        image_base64: formData.image || null,
        limit: 10,
        similarity_threshold: 0.0
      };

      if (Object.keys(conditionsObj).length > 0) {
        searchPayload.conditions = conditionsObj;
      }

      console.log('Sending search payload:', searchPayload);

      const response = await fetch('/api/v1/search/direct', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token') || ''}`
        },
        body: JSON.stringify(searchPayload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Search Error:', JSON.stringify(errorData, null, 2));
        throw new Error(JSON.stringify(errorData));
      }

      const results = await response.json();
      onSearchResults(results);
    } catch (err) {
      console.error('Search error:', err);
      setError(err.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDesign = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const designRequest = {
        text: formData.text,
        sequence: formData.sequence || undefined,
        conditions: {},
        num_variants: parseInt(formData.numVariants) || 3,
        goal: 'optimize_protocol',
      };

      if (formData.organism) designRequest.conditions.organism = formData.organism;
      if (formData.temperature) designRequest.conditions.temperature = parseFloat(formData.temperature);
      if (formData.ph) designRequest.conditions.ph = parseFloat(formData.ph);

      if (Object.keys(designRequest.conditions).length === 0) {
        delete designRequest.conditions;
      }

      const results = await designAPI.generateVariants(designRequest);
      onDesignResults(results);
    } catch (err) {
      const errorDetail = err.response?.data?.detail;
      setError(formatValidationErrors(errorDetail) || 'Design generation failed');
    } finally {
      setLoading(false);
    }
  };

  const renderFacetSelect = (name, label, facetData) => {
    const counts = facetData?.counts || facetData?.facets || {};
    const options = Object.entries(counts);
    if (!options.length) {
      return (
        <div style={styles.field}>
          <input
            type="text"
            name={name}
            value={formData[name]}
            onChange={handleChange}
            placeholder={label}
            style={styles.input}
          />
        </div>
      );
    }
    return (
      <div style={styles.field}>
        <select name={name} value={formData[name]} onChange={handleChange} style={styles.input}>
          <option value="">{label} (all)</option>
          {options.map(([value, count]) => (
            <option key={value} value={value}>
              {value} ({count})
            </option>
          ))}
        </select>
      </div>
    );
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2 style={styles.title}>
          {mode === 'search' ? (
            <><Search size={24} /> Multimodal Search</>
          ) : (
            <><Beaker size={24} /> AI-Assisted Design</>
          )}
        </h2>
        <div style={styles.modeSwitcher}>
          <button
            style={mode === 'search' ? styles.modeButtonActive : styles.modeButton}
            onClick={() => setMode('search')}
          >
            <Search size={18} /> Search
          </button>
          <button
            style={mode === 'design' ? styles.modeButtonActive : styles.modeButton}
            onClick={() => setMode('design')}
          >
            <Beaker size={18} /> Design
          </button>
        </div>
      </div>

      {mode === 'search' && (
        <div style={styles.searchModeSelector}>
          <label style={styles.searchModeLabel}>Search Mode:</label>
          <div style={styles.searchModeButtons}>
            <button type="button" style={searchMode === 'multimodal' ? styles.searchModeActive : styles.searchModeButton} onClick={() => setSearchMode('multimodal')}>Multimodal</button>
            <button type="button" style={searchMode === 'text' ? styles.searchModeActive : styles.searchModeButton} onClick={() => setSearchMode('text')}>Text Only</button>
            <button type="button" style={searchMode === 'image' ? styles.searchModeActive : styles.searchModeButton} onClick={() => setSearchMode('image')}><Image size={16} /> Image Only</button>
          </div>
        </div>
      )}

      <form onSubmit={searchMode === 'image' ? handleImageSearch : (mode === 'search' ? handleSearch : handleDesign)} style={styles.form}>
        {/* Text Input - hidden for image-only mode */}
        {(searchMode !== 'image' || mode === 'design') && (
          <div style={styles.field}>
            <label style={styles.label}><Search size={18} /> Description</label>
            <textarea
              name="text"
              value={formData.text}
              onChange={handleChange}
              placeholder={mode === 'search' ? 'Describe the experiment you are looking for...' : 'Describe the experiment you want to optimize...'}
              style={styles.textarea}
              rows={4}
              required
            />
          </div>
        )}

        {/* Sequence Input - hidden for image-only and text-only modes */}
        {(searchMode === 'multimodal' || mode === 'design') && (
          <div style={styles.field}>
            <label style={styles.label}><Dna size={18} /> Biological Sequence (optional)</label>
            <input type="text" name="sequence" value={formData.sequence} onChange={handleChange} placeholder="ATGGCTAGCAAAGGAGAAG..." style={styles.input} />
            <div style={styles.hint}>DNA, RNA, or protein sequence</div>
          </div>
        )}

        {/* Image Input - shown for multimodal and image modes */}
        {(searchMode === 'multimodal' || searchMode === 'image') && mode === 'search' && (
          <div style={styles.field}>
            <label style={styles.label}><Image size={18} /> Experiment Image {searchMode === 'image' ? '(required)' : '(optional)'}</label>
            <input type="file" accept="image/*" onChange={handleImageUpload} style={styles.fileInput} required={searchMode === 'image'} />
            {formData.image && (
              <div style={styles.imagePreview}>
                <img src={formData.image} alt="Preview" style={styles.previewImage} />
              </div>
            )}
            <div style={styles.hint}>Upload gel images, microscopy results, or experimental photos</div>
          </div>
        )}

        {/* Experimental Conditions - hidden for image-only mode */}
        {(searchMode !== 'image' || mode === 'design') && (
        <div style={styles.conditionsSection}>
          <label style={styles.label}><Sliders size={18} /> Experimental Conditions (optional)</label>
          <div style={styles.conditionsGrid}>
            {facets?.organisms
              ? renderFacetSelect('organism', 'Organism', facets.organisms)
              : (
                <div style={styles.field}>
                  <input type="text" name="organism" value={formData.organism} onChange={handleChange} placeholder="Organism (e.g., human, ecoli)" style={styles.input} />
                </div>
              )
            }
            <div style={styles.field}>
              <input type="number" name="temperature" value={formData.temperature} onChange={handleChange} placeholder="Temperature (C)" style={styles.input} step="0.1" />
            </div>
            <div style={styles.field}>
              <input type="number" name="ph" value={formData.ph} onChange={handleChange} placeholder="pH" style={styles.input} step="0.1" min="0" max="14" />
            </div>
          </div>
        </div>
        )}

        {/* Advanced Filters Toggle */}
        {mode === 'search' && searchMode !== 'image' && (
          <div style={styles.advancedToggle}>
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              style={styles.advancedToggleButton}
            >
              <Filter size={16} />
              Advanced Filters
              {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
          </div>
        )}

        {/* Advanced Filters */}
        {showAdvanced && mode === 'search' && searchMode !== 'image' && (
          <div style={styles.advancedFilters}>
            <div style={styles.conditionsGrid}>
              {facets?.assays
                ? renderFacetSelect('assay', 'Assay Type', facets.assays)
                : (
                  <div style={styles.field}>
                    <input type="text" name="assay" value={formData.assay} onChange={handleChange} placeholder="Assay (e.g., western_blot, qPCR)" style={styles.input} />
                  </div>
                )
              }
              {facets?.sources
                ? renderFacetSelect('source', 'Source', facets.sources)
                : (
                  <div style={styles.field}>
                    <input type="text" name="source" value={formData.source} onChange={handleChange} placeholder="Source (e.g., pubmed)" style={styles.input} />
                  </div>
                )
              }
              <div style={styles.field}>
                <input type="text" name="type" value={formData.type} onChange={handleChange} placeholder="Type (e.g., expression_analysis)" style={styles.input} />
              </div>
            </div>
            <div style={styles.checkboxRow}>
              <label style={styles.checkboxLabel}>
                <input type="checkbox" name="successOnly" checked={formData.successOnly} onChange={handleChange} />
                Successful experiments only
              </label>
            </div>
          </div>
        )}

        {/* Design-specific: Number of variants */}
        {mode === 'design' && (
          <div style={styles.field}>
            <label style={styles.label}><Beaker size={18} /> Number of Variants</label>
            <input type="number" name="numVariants" value={formData.numVariants} onChange={handleChange} style={styles.input} min="1" max="10" />
            <div style={styles.hint}>Generate 1-10 design variants</div>
          </div>
        )}

        {/* Error Display */}
        {error && <div style={styles.error}>{error}</div>}

        {/* Submit Button */}
        <button type="submit" disabled={loading} style={styles.submitButton}>
          {loading ? 'Processing...' : searchMode === 'image' ? (
            <><Image size={20} /> Search by Image</>
          ) : mode === 'search' ? (
            <><Search size={20} /> Search Experiments</>
          ) : (
            <><Beaker size={20} /> Generate Variants</>
          )}
        </button>
      </form>

      <div style={styles.infoBox}></div>
    </div>
  );
}

const styles = {
  container: { maxWidth: '900px', margin: '0 auto' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' },
  title: { fontSize: '1.8rem', color: '#333', display: 'flex', alignItems: 'center', gap: '0.5rem' },
  modeSwitcher: { display: 'flex', gap: '0.5rem' },
  modeButton: { padding: '0.5rem 1rem', border: '2px solid #0a1131', background: '#fff', color: '#0a1131', borderRadius: '8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: '500', transition: 'all 0.2s' },
  modeButtonActive: { padding: '0.5rem 1rem', border: '2px solid #0a1131', background: '#0a1131', color: '#fff', borderRadius: '8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: '600' },
  form: { background: '#f9fafb', padding: '2rem', borderRadius: '12px', border: '1px solid #e5e7eb' },
  field: { marginBottom: '1.5rem' },
  label: { display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: '#374151', fontWeight: '600', fontSize: '0.95rem' },
  input: { width: '100%', padding: '0.75rem', border: '2px solid #d1d5db', borderRadius: '8px', fontSize: '1rem', transition: 'border 0.2s', boxSizing: 'border-box' },
  textarea: { width: '100%', padding: '0.75rem', border: '2px solid #d1d5db', borderRadius: '8px', fontSize: '1rem', resize: 'vertical', fontFamily: 'inherit', boxSizing: 'border-box' },
  hint: { marginTop: '0.25rem', fontSize: '0.85rem', color: '#6b7280' },
  conditionsSection: { marginBottom: '1.5rem' },
  conditionsGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginTop: '0.5rem' },
  error: { padding: '1rem', background: '#fee2e2', border: '1px solid #fecaca', borderRadius: '8px', color: '#991b1b', marginBottom: '1rem', whiteSpace: 'pre-line' },
  submitButton: { width: '100%', padding: '1rem', background: 'linear-gradient(135deg, #0a1131 0%, #0a1131 100%)', color: '#fff', border: 'none', borderRadius: '8px', fontSize: '1.1rem', fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', transition: 'transform 0.2s, box-shadow 0.2s' },
  infoBox: { marginTop: '2rem', padding: '1.5rem', background: '#eff6ff', border: '1px solid #dbeafe', borderRadius: '12px' },
  searchModeSelector: { marginBottom: '1.5rem', padding: '1rem', background: '#f3f4f6', borderRadius: '8px', border: '1px solid #e5e7eb' },
  searchModeLabel: { display: 'block', marginBottom: '0.5rem', color: '#374151', fontWeight: '600', fontSize: '0.95rem' },
  searchModeButtons: { display: 'flex', gap: '0.5rem', flexWrap: 'wrap' },
  searchModeButton: { padding: '0.5rem 1rem', border: '2px solid #d1d5db', background: '#fff', color: '#374151', borderRadius: '6px', cursor: 'pointer', fontSize: '0.9rem', fontWeight: '500', transition: 'all 0.2s', display: 'flex', alignItems: 'center', gap: '0.3rem' },
  searchModeActive: { padding: '0.5rem 1rem', border: '2px solid #0a1131', background: '#0a1131', color: '#fff', borderRadius: '6px', cursor: 'pointer', fontSize: '0.9rem', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.3rem' },
  fileInput: { width: '100%', padding: '0.75rem', border: '2px solid #d1d5db', borderRadius: '8px', fontSize: '1rem', background: '#fff', boxSizing: 'border-box' },
  imagePreview: { marginTop: '1rem', textAlign: 'center' },
  previewImage: { maxWidth: '100%', maxHeight: '200px', borderRadius: '8px', border: '1px solid #e5e7eb' },
  advancedToggle: { marginBottom: '1rem' },
  advancedToggleButton: { display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem', background: 'transparent', border: '1px solid #d1d5db', borderRadius: '6px', cursor: 'pointer', color: '#374151', fontSize: '0.9rem', fontWeight: '500' },
  advancedFilters: { marginBottom: '1.5rem', padding: '1rem', background: '#f0f4ff', borderRadius: '8px', border: '1px dashed #93a5cf' },
  checkboxRow: { marginTop: '0.75rem', display: 'flex', gap: '1.5rem', flexWrap: 'wrap' },
  checkboxLabel: { display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: '#374151', cursor: 'pointer' },
};
