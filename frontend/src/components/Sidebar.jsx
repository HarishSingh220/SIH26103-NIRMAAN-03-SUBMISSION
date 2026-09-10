import { useState } from "react";
import { NavLink } from "react-router-dom";
import "./Sidebar.css";

import {
  DashboardIcon,
  ProjectsIcon,
  RiskIcon,
  InfoIcon,
  AIIcon,
  SettingsIcon,
  ChevronIcon,
} from "./Icons";

function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  const linkClass = ({ isActive }) =>
    isActive ? "sidebar-item active" : "sidebar-item";

  return (
    <aside className={collapsed ? "sidebar collapsed" : "sidebar"}>
      <button
        className="sidebar-toggle"
        onClick={() => setCollapsed(!collapsed)}
        aria-label="Toggle sidebar"
      >
        <ChevronIcon />
      </button>

      <div className="sidebar-menu">
        <NavLink to="/" end className={linkClass}>
          <DashboardIcon />
          <span>Dashboard</span>
        </NavLink>

        <NavLink to="/project-overview" className={linkClass}>
          <ProjectsIcon />
          <span>Project Overview</span>
        </NavLink>

        <NavLink to="/risk-analysis" className={linkClass}>
          <RiskIcon />
          <span>Project Risk Analysis</span>
        </NavLink>

        <NavLink to="/about" className={linkClass}>
          <InfoIcon />
          <span>About</span>
        </NavLink>

        <NavLink to="/ai-assistant" className={linkClass}>
          <AIIcon />
          <span>AI Assistant</span>
        </NavLink>
      </div>

      <div className="sidebar-bottom">
        <button className="sidebar-item">
          <SettingsIcon />
          <span>Settings</span>
        </button>
      </div>
    </aside>
  );
}

export default Sidebar;
