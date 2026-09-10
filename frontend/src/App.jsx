import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Home from "./pages/Home";
import ProjectOverview from "./pages/ProjectOverview";
import RiskAnalysis from "./pages/RiskAnalysis";
import About from "./pages/About";
import AIAssistant from "./pages/AIAssistant";
import Sidebar from "./components/Sidebar";
import Login from "./pages/Login";
import Register from "./pages/Register";

/** Returns true if a valid login token exists in localStorage. */
function isLoggedIn() {
  return Boolean(localStorage.getItem("nirmaan_token"));
}

/**
 * Wraps a protected page: redirects to /login when the user is not logged in.
 * Renders children unchanged when the user IS logged in.
 */
function ProtectedRoute({ children }) {
  return isLoggedIn() ? children : <Navigate to="/login" replace />;
}

/**
 * Wraps a public-only page (login / register): redirects to / when the user
 * is already logged in so they never see the auth screens again.
 */
function PublicRoute({ children }) {
  return isLoggedIn() ? <Navigate to="/" replace /> : children;
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Auth pages — redirect to dashboard if already logged in */}
        <Route path="/login"    element={<PublicRoute><Login /></PublicRoute>} />
        <Route path="/register" element={<PublicRoute><Register /></PublicRoute>} />

        {/* Main application — redirect to login if not logged in */}
        <Route
          path="/*"
          element={
            <ProtectedRoute>
              <div className="app-shell">
                <Sidebar />

                <main className="app-main">
                  <Routes>
                    <Route path="/"                  element={<Home />} />
                    <Route path="/project-overview"  element={<ProjectOverview />} />
                    <Route path="/risk-analysis"     element={<RiskAnalysis />} />
                    <Route path="/about"             element={<About />} />
                    <Route path="/ai-assistant"      element={<AIAssistant />} />
                  </Routes>
                </main>
              </div>
            </ProtectedRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;

