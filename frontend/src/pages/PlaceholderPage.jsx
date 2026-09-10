import Navbar from "../components/Navbar";
import "./Home.css";
import "./PlaceholderPage.css";

function PlaceholderPage({ eyebrow, title, subtitle, icon }) {
  return (
    <>
      <Navbar />

      <div className="home">
        <div className="placeholder-page">
          <span className="placeholder-icon" aria-hidden="true">
            {icon}
          </span>
          <p className="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
          <p className="subtitle">{subtitle}</p>
          <span className="placeholder-badge">Coming soon</span>
        </div>
      </div>
    </>
  );
}

export default PlaceholderPage;
