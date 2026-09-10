import { useState } from "react";
import { Link } from "react-router-dom";

function LoginForm({ onLogin }) {
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const [formData, setFormData] = useState({
    username: "",
    password: "",
  });

  const [error, setError] = useState("");

  function handleChange(e) {
    const { name, value } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));

    setError("");
  }

  async function handleSubmit(e) {
    e.preventDefault();

    if (!formData.username.trim() || !formData.password.trim()) {
      setError("Please enter your username and password.");
      return;
    }

    setLoading(true);
    setError("");
    // onLogin receives (formData, setError) so the parent can surface API errors
    await onLogin(formData, setError);
    setLoading(false);
  }

  return (
    <section className="login-panel">
      <div className="login-card">

        <div className="tricolor-line">
          <span></span>
          <span></span>
          <span></span>
        </div>

        <h2>Welcome Back</h2>

        <p className="login-subtitle">
          Login to your NIRMAAN account
        </p>

        <form onSubmit={handleSubmit}>

          <div className="input-group">
            <label>Email or Username</label>

            <div className="input-wrapper">
              <span className="input-icon">⌑</span>

              <input
                type="text"
                name="username"
                value={formData.username}
                onChange={handleChange}
                placeholder="Enter your email or username"
              />
            </div>
          </div>

          <div className="input-group">
            <label>Password</label>

            <div className="input-wrapper">
              <span className="input-icon">⌑</span>

              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Enter your password"
              />

              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? "◉" : "○"}
              </button>
            </div>
          </div>

          {error && (
            <p className="login-error">
              {error}
            </p>
          )}

          <div className="login-options">

            <label className="remember">
              <input type="checkbox" />
              <span>Remember me</span>
            </label>

            <button
              type="button"
              className="forgot"
              onClick={() => alert("Forgot password feature coming soon.")}
            >
              Forgot password?
            </button>

          </div>

          <button
            type="submit"
            className="login-button"
            disabled={loading}
          >
            <span>{loading ? "Logging in…" : "Login"}</span>
            <span className="login-arrow">→</span>
          </button>

        </form>

        <div className="divider">
          <span></span>
          <p>OR</p>
          <span></span>
        </div>

          <p className="register-text">
            Don't have an account?
           <Link to="/register">
             Register
            </Link>
          </p>
      </div>
    </section>
  );
}

export default LoginForm;