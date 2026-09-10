import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Navbar.css";

import {
  SearchIcon,
  BellIcon,
  ChevronIcon,
  ProfileIcon,
  SettingsIcon,
  LogoutIcon
} from "./Icons";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function Navbar() {
  const [profileOpen, setProfileOpen] = useState(false);
  const navigate = useNavigate();

  // Read user info from localStorage (set at login time)
  const storedUser = (() => {
    try {
      const raw = localStorage.getItem("nirmaan_user");
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  })();

  const displayName = storedUser?.fullName || "User";
  const displayRole = storedUser?.role || "Project Officer";
  const avatarLetter = displayName.charAt(0).toUpperCase();

  async function handleLogout() {
    setProfileOpen(false);
    // Best-effort server logout (token is stateless; clearing localStorage is sufficient)
    const token = localStorage.getItem("nirmaan_token");
    if (token) {
      fetch(`${API_BASE}/api/auth/logout`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      }).catch(() => {/* ignore network errors on logout */});
    }
    localStorage.removeItem("nirmaan_token");
    localStorage.removeItem("nirmaan_user");
    navigate("/login");
  }

  return (
    <nav className="navbar">

      {/* Brand */}
      <div className="navbar-brand">
        <div className="logo">N</div>

        <div className="brand-text">
          <h2>NIRMAAN</h2>
          <span>Project Monitoring</span>
        </div>
      </div>

      {/* Actions */}
      <div className="navbar-actions">

        {/* Search */}
        <button className="search-button" type="button">
          <SearchIcon />
          <span className="search-label">Search projects</span>
          <span className="search-shortcut">Ctrl K</span>
        </button>

        {/* Notifications */}
        <button
          className="icon-button notification-button"
          type="button"
          aria-label="Notifications"
        >
          <BellIcon />
          <span className="notification-dot"></span>
        </button>

        {/* Profile */}
        <div className="profile-wrapper">

          <button
            className="profile-button"
            type="button"
            onClick={() => setProfileOpen(!profileOpen)}
          >
            <span className="avatar">{avatarLetter}</span>

            <span className="profile-info">
              <strong>{displayName}</strong>
              <small>{displayRole}</small>
            </span>

            <ChevronIcon />
          </button>

          {/* Profile Dropdown */}
          {profileOpen && (
            <div className="profile-dropdown">

              <button
                type="button"
                className="dropdown-item"
              >
                <ProfileIcon />
                Profile
              </button>

              <button
                type="button"
                className="dropdown-item"
              >
                <SettingsIcon />
                Settings
              </button>

              <div className="dropdown-divider"></div>

              <button
                type="button"
                className="dropdown-item logout-item"
                onClick={handleLogout}
              >
                <LogoutIcon />
                Log out
              </button>

            </div>
          )}

        </div>

      </div>

    </nav>
  );
}

export default Navbar;