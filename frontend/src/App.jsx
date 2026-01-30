/**
 * App Component
 * Main application component
 */
import { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import { authAPI } from './services/api';

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isRegisterMode, setIsRegisterMode] = useState(false);
  const [loginForm, setLoginForm] = useState({ email: '', password: '' });
  const [registerForm, setRegisterForm] = useState({ email: '', password: '', fullName: '' });
  const [error, setError] = useState(null);

  // Set initial mode from URL path
  useEffect(() => {
    const path = window.location.pathname;
    if (path === '/register') setIsRegisterMode(true);
    checkAuth();
  }, []);

  // Update URL when view changes
  useEffect(() => {
    if (isLoading) return;
    if (isAuthenticated) {
      window.history.replaceState(null, '', '/dashboard');
    } else if (isRegisterMode) {
      window.history.replaceState(null, '', '/register');
    } else {
      window.history.replaceState(null, '', '/login');
    }
  }, [isAuthenticated, isRegisterMode, isLoading]);

  const checkAuth = async () => {
    const token = localStorage.getItem('access_token');
    if (token) {
      try {
        await authAPI.getMe();
        setIsAuthenticated(true);
      } catch {
        localStorage.removeItem('access_token');
        setIsAuthenticated(false);
      }
    }
    setIsLoading(false);
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

  const handleLogin = async (e) => {
    e.preventDefault();
    setError(null);

    try {
      await authAPI.login(loginForm.email, loginForm.password);
      setIsAuthenticated(true);
    } catch (err) {
      const errorDetail = err.response?.data?.detail;
      setError(formatValidationErrors(errorDetail) || 'Login failed');
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError(null);

    try {
      await authAPI.register(registerForm.email, registerForm.password, registerForm.fullName);
      // After successful registration, switch to login mode
      setIsRegisterMode(false);
      setError('Registration successful! Please login.');
      setLoginForm({ email: registerForm.email, password: '' });
    } catch (err) {
      const errorDetail = err.response?.data?.detail;
      setError(formatValidationErrors(errorDetail) || 'Registration failed');
    }
  };

  const handleLogout = () => {
    authAPI.logout();
    setIsAuthenticated(false);
  };

  if (isLoading) {
    return (
      <div style={styles.loading}>
        <div style={styles.loadingSpinner} />
        <p>Loading BIOMEMORY...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div style={styles.loginContainer}>
        <div style={styles.loginCard}>
          <h1 style={styles.loginTitle}> BIOMEMORY</h1>
          <p style={styles.loginSubtitle}>Biological Design & Discovery Intelligence</p>

          {isRegisterMode ? (
            <>
              <h2 style={styles.formTitle}>Create Account</h2>
              <form onSubmit={handleRegister} style={styles.loginForm}>
                <input
                  type="text"
                  placeholder="Full Name"
                  value={registerForm.fullName}
                  onChange={(e) => setRegisterForm({ ...registerForm, fullName: e.target.value })}
                  style={styles.loginInput}
                  required
                />
                <input
                  type="email"
                  placeholder="Email"
                  value={registerForm.email}
                  onChange={(e) => setRegisterForm({ ...registerForm, email: e.target.value })}
                  style={styles.loginInput}
                  required
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={registerForm.password}
                  onChange={(e) => setRegisterForm({ ...registerForm, password: e.target.value })}
                  style={styles.loginInput}
                  required
                />

                {error && <div style={styles.loginError}>{error}</div>}

                <button type="submit" style={styles.loginButton}>
                  Register
                </button>
              </form>
            </>
          ) : (
            <>
              <h2 style={styles.formTitle}>Welcome Back</h2>
              <form onSubmit={handleLogin} style={styles.loginForm}>
                <input
                  type="email"
                  placeholder="Email"
                  value={loginForm.email}
                  onChange={(e) => setLoginForm({ ...loginForm, email: e.target.value })}
                  style={styles.loginInput}
                  required
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={loginForm.password}
                  onChange={(e) => setLoginForm({ ...loginForm, password: e.target.value })}
                  style={styles.loginInput}
                  required
                />

                {error && <div style={styles.loginError}>{error}</div>}

                <button type="submit" style={styles.loginButton}>
                  Login
                </button>
              </form>
            </>
          )}

          <div style={styles.modeToggle}>
            <button
              type="button"
              onClick={() => {
                setIsRegisterMode(!isRegisterMode);
                setError(null);
              }}
              style={styles.toggleButton}
            >
              {isRegisterMode ? 'Already have an account? Login' : 'Need an account? Register'}
            </button>
          </div>

          <div style={styles.loginDemo}>
            
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.app}>
      <Dashboard />
      <button onClick={handleLogout} style={styles.logoutButton}>
        Logout
      </button>
    </div>
  );
}

const styles = {
  app: {
    position: 'relative',
  },
  loading: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #0a1131 0%, #0a1131 100%)',
    color: '#fff',
  },
  loadingSpinner: {
    width: '50px',
    height: '50px',
    border: '5px solid rgba(22, 13, 70, 0.3)',
    borderTop: '5px solid #fff',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  loginContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '100vh',
    background: 'url(/background.jpg) no-repeat center center fixed',
    backgroundSize: 'cover',
    padding: '2rem',
  },
  loginCard: {
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(20px)',
    padding: '3rem',
    borderRadius: '20px',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    maxWidth: '400px',
    width: '100%',
    border: '1px solid rgba(15, 7, 63, 0.2)',
  },
  loginTitle: {
    fontSize: '2.5rem',
    color: '#0a1131',
    textAlign: 'center',
    marginBottom: '0.5rem',
    textShadow: '0 2px 4px rgba(12, 16, 53, 0.1)',
  },
  loginSubtitle: {
    fontSize: '0.95rem',
    color: '#0a1131',
    textAlign: 'center',
    marginBottom: '2rem',
    fontWeight: '500',
  },
  loginForm: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  loginInput: {
    padding: '1rem',
    border: '2px solid #e5e7eb',
    borderRadius: '8px',
    fontSize: '1rem',
    transition: 'border 0.2s',
  },
  loginError: {
    padding: '0.75rem',
    background: '#42538b',
    border: '1px solid #0a1131',
    borderRadius: '8px',
    color: '#0a1131',
    fontSize: '0.9rem',
    whiteSpace: 'pre-line',
  },
  loginButton: {
    padding: '1rem',
    background: 'linear-gradient(135deg, #0a1131 0%, #0a1131 100%)',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1.1rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'transform 0.2s',
  },
  loginDemo: {
    marginTop: '2rem',
    padding: '1rem',
    background: '#f9fafb',
    borderRadius: '8px',
    fontSize: '0.85rem',
    color: '#6b7280',
    textAlign: 'center',
  },
  formTitle: {
    fontSize: '1.5rem',
    color: '#0a1131',
    textAlign: 'center',
    marginBottom: '1.5rem',
    fontWeight: '600',
  },
  modeToggle: {
    marginTop: '1.5rem',
    textAlign: 'center',
  },
  toggleButton: {
    background: 'none',
    border: 'none',
    color: '#0a1131',
    cursor: 'pointer',
    fontSize: '0.9rem',
    textDecoration: 'underline',
    padding: '0.5rem',
    transition: 'color 0.2s',
    ':hover': {
      color: '#0a1131',
    },
  },
  loginHint: {
    fontSize: '0.75rem',
    fontStyle: 'italic',
    marginTop: '0.5rem',
  },
  logoutButton: {
    position: 'fixed',
    top: '1rem',
    right: '1rem',
    padding: '0.5rem 1rem',
    background: 'rgba(255, 255, 255, 0.2)',
    color: '#fff',
    border: '1px solid rgba(255, 255, 255, 0.3)',
    borderRadius: '8px',
    cursor: 'pointer',
    fontWeight: '600',
    backdropFilter: 'blur(10px)',
  },
};
