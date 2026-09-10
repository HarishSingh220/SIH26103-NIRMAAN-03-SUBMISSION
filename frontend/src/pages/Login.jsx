import { useNavigate } from "react-router-dom";
import "./Login.css";

import LoginVisual from "../components/LoginVisual.jsx";
import LoginForm from "../components/LoginForm.jsx";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function Login() {
  const navigate = useNavigate();

  async function handleLogin(credentials, setError) {
    try {
      const response = await fetch(`${API_BASE}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: credentials.username,
          password: credentials.password,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        const msg = data?.detail?.message || data?.detail || "Login failed. Please check your credentials.";
        setError(msg);
        return;
      }

      // Store token and user info
      localStorage.setItem("nirmaan_token", data.data.token);
      localStorage.setItem("nirmaan_user", JSON.stringify(data.data.user));

      // Navigate to home
      navigate("/");
    } catch {
      setError("Unable to reach the server. Please ensure the backend is running.");
    }
  }

  return (
    <div className="login-page">

      <div className="network-background">
        <span className="floating-dot dot-1"></span>
        <span className="floating-dot dot-2"></span>
        <span className="floating-dot dot-3"></span>
        <span className="floating-dot dot-4"></span>
        <span className="floating-dot dot-5"></span>
        <span className="floating-dot dot-6"></span>
      </div>

      <LoginVisual />

      <LoginForm onLogin={handleLogin} />

    </div>
  );
}

export default Login;