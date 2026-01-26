/**
 * API Service for BIOMEMORY Backend
 */
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests if available
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  register: async (email, password, fullName) => {
    const response = await apiClient.post('/auth/register', {
      email,
      password,
      full_name: fullName,
    });
    return response.data;
  },

  login: async (email, password) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);

    const response = await apiClient.post('/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    if (response.data.access_token) {
      localStorage.setItem('access_token', response.data.access_token);
    }

    return response.data;
  },

  logout: () => {
    localStorage.removeItem('access_token');
    window.location.href = '/login';
  },

  getMe: async () => {
    const response = await apiClient.get('/auth/me');
    return response.data;
  },
};

// Experiments API
export const experimentsAPI = {
  upload: async (experimentData) => {
    const response = await apiClient.post('/experiments/upload', experimentData);
    return response.data;
  },

  list: async (limit = 20, offset = 0, successOnly = null) => {
    const params = { limit, offset };
    if (successOnly !== null) {
      params.success_only = successOnly;
    }
    const response = await apiClient.get('/experiments/', { params });
    return response.data;
  },

  get: async (experimentId) => {
    const response = await apiClient.get(`/experiments/${experimentId}`);
    return response.data;
  },

  delete: async (experimentId) => {
    const response = await apiClient.delete(`/experiments/${experimentId}`);
    return response.data;
  },

  getStats: async () => {
    const response = await apiClient.get('/experiments/stats/summary');
    return response.data;
  },
};

// Search API
export const searchAPI = {
  search: async (query) => {
    const response = await apiClient.post('/search/', query);
    return response.data;
  },

  searchByImage: async (imageBase64, options = {}) => {
    const formData = new FormData();
    formData.append('image_base64', imageBase64);
    formData.append('limit', options.limit || 10);
    formData.append('include_failures', options.include_failures !== false);

    const response = await apiClient.post('/search/image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  advancedSearch: async (query, filters = {}) => {
    const params = new URLSearchParams();
    if (filters.include_failures !== undefined) {
      params.append('include_failures', filters.include_failures);
    }
    if (filters.min_success_rate) {
      params.append('min_success_rate', filters.min_success_rate);
    }
    if (filters.organism_filter) {
      params.append('organism_filter', filters.organism_filter);
    }

    const response = await apiClient.post(`/search/advanced?${params}`, query);
    return response.data;
  },

  getSuggestions: async (query, limit = 5) => {
    const response = await apiClient.get('/search/suggestions', {
      params: { query, limit },
    });
    return response.data;
  },
};

// Design API
export const designAPI = {
  generateVariants: async (designRequest) => {
    const response = await apiClient.post('/design/variants', designRequest);
    return response.data;
  },

  optimize: async (experimentId, goal = 'increase_yield') => {
    const params = new URLSearchParams({ goal });
    const response = await apiClient.post(`/design/optimize?experiment_id=${experimentId}&${params}`);
    return response.data;
  },

  troubleshoot: async (failedExperimentId) => {
    const params = new URLSearchParams({ failed_experiment_id: failedExperimentId });
    const response = await apiClient.post(`/design/troubleshoot?${params}`);
    return response.data;
  },

  getTemplates: async (category = null) => {
    const params = category ? { category } : {};
    const response = await apiClient.get('/design/templates', { params });
    return response.data;
  },
};

// Health API
export const healthAPI = {
  check: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },
};

export default apiClient;
