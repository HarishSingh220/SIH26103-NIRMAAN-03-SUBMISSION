import { useNavigate } from "react-router-dom";
import "./Login.css";
import LoginVisual from "../components/LoginVisual.jsx";
import RegisterForm from "../components/RegisterForm.jsx";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function Register() {
  const navigate = useNavigate();

  async function handleRegister(credentials, setError) {
    try {
      const response = await fetch(`${API_BASE}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fullName: credentials.fullName,
          email: credentials.email,
          password: credentials.password,
          confirmPassword: credentials.confirmPassword,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        const msg = data?.detail?.message || data?.detail || "Registration failed. Please try again.";
        setError(msg);
        return;
      }

      // Navigate to login after successful registration
      navigate("/login", { state: { registered: true } });
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

      <RegisterForm onRegister={handleRegister} />

    </div>
  );
}

export default Register;