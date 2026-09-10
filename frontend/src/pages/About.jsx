import "./About.css";
import Navbar from "../components/Navbar";

function About() {
  return (
    <>
      <Navbar />

      <div className="about-page">

      {/* Hero Section */}
      <section className="about-hero">
        <div className="about-badge">
          <span className="badge-dot"></span>
          NATIONAL PROJECT MONITORING PLATFORM
        </div>
        <div className="about-heading-line">
          <span className="about-tricolor-line">
            <i /><i /><i />
          </span>
        </div>
        <h1>
          Building a <span>Better India</span><br />
          Through Smarter Monitoring
        </h1>

        <p>
          NIRMAAN is a project monitoring platform designed to bring
          transparency, intelligence, and efficiency to large-scale
          infrastructure and development projects across India.
        </p>
      </section>

      {/* About Card */}
      <section className="about-content">

        <div className="about-card about-main-card">
          <div className="card-icon">N</div>

          <div>
            <h2>What is NIRMAAN?</h2>

            <p>
              NIRMAAN is a centralized project monitoring and decision-support
              platform that helps stakeholders track project progress,
              identify potential risks, and make informed decisions.
            </p>

            <p>
              By bringing project information together in one place, NIRMAAN
              enables a clearer view of ongoing projects and helps teams
              respond proactively to emerging challenges.
            </p>
          </div>
        </div>

        {/* Feature Cards */}
        <div className="about-grid">

          <div className="about-card feature-card">
            <div className="feature-icon">◈</div>
            <h3>Centralized Monitoring</h3>
            <p>
              Monitor projects, progress, and key information from a
              unified platform.
            </p>
          </div>

          <div className="about-card feature-card">
            <div className="feature-icon">◉</div>
            <h3>Risk Intelligence</h3>
            <p>
              Identify potential project risks early and support
              proactive decision-making.
            </p>
          </div>

          <div className="about-card feature-card">
            <div className="feature-icon">↗</div>
            <h3>Data-Driven Decisions</h3>
            <p>
              Turn project data into meaningful insights for better
              planning and execution.
            </p>
          </div>

          <div className="about-card feature-card">
            <div className="feature-icon">✓</div>
            <h3>Greater Transparency</h3>
            <p>
              Improve visibility into project performance and
              strengthen accountability.
            </p>
          </div>

        </div>

        {/* AI Approach Section */}
        <div className="about-card about-main-card approach-card">
          <div className="card-icon">AI</div>

          <div>
            <span className="section-label">HOW NIRMAAN WORKS</span>

            <h2>Intelligent Infrastructure Monitoring &amp; Risk Analytics</h2>

            <p>
              NIRMAAN is an AI-powered infrastructure project monitoring and
              early-warning platform designed to help identify potential cost
              overruns, time delays, implementation risks, and unusual
              project behaviour before they become critical.
            </p>

            <p>
              By combining historical project data with Machine Learning,
              predictive analytics, anomaly detection, explainable AI and
              Large Language Models, NIRMAAN transforms traditional project
              monitoring from simply understanding what has happened into
              predicting what is likely to happen next.
            </p>

            <p>
              It provides project-level risk analysis through an interactive
              dashboard, helping monitoring authorities, ministries and
              implementing agencies make faster, data-driven and proactive
              decisions.
            </p>
          </div>
        </div>

        {/* Vision Quote */}
        <div className="quote-banner">
          <span className="quote-mark">&ldquo;</span>
          <p>
            From Monitoring What Happened to<br />
            Predicting What Happens Next.
          </p>
        </div>

        {/* Vision Section */}
        <div className="vision-card">
          <div className="card-icon vision-icon">V</div>

          <div>
            <span className="section-label">OUR VISION</span>

            <h2>
              Smarter Projects.
              <br />
              Stronger Infrastructure.
            </h2>

            <p>
              Our vision is to create a technology-driven ecosystem where
              project monitoring becomes proactive, transparent, and
              intelligent—helping India build faster and better.
            </p>

            <p>
              NIRMAAN aims to bridge the gap between project data and
              actionable decisions, so that potential risks are identified
              early, understood clearly, and addressed in time—enabling
              stakeholders to move from reactive monitoring to predictive
              and preventive intervention.
            </p>
          </div>

          <div className="vision-mark">
            <span>भारत</span>
            <small>BUILDING THE FUTURE</small>
          </div>
        </div>

        {/* What NIRMAAN Does */}
        <div className="about-section-heading">
          <span className="section-label">WHAT NIRMAAN DOES</span>
          <h2>Three Questions, One Clear Picture</h2>
          <p>
            NIRMAAN analyses project information and answers three key
            questions to build a complete view of project health.
          </p>
        </div>

        <div className="process-grid">

          <div className="about-card process-card">
            <div className="process-number">01</div>
            <h3>Is the project at risk?</h3>
            <p>Our classification models identify the possibility of:</p>
            <ul className="process-list">
              <li>Cost overrun</li>
              <li>Time overrun</li>
              <li>Overall project risk</li>
            </ul>
          </div>

          <div className="about-card process-card">
            <div className="process-number">02</div>
            <h3>How severe could it be?</h3>
            <p>
              Regression models estimate the potential magnitude of cost or
              time overrun, helping stakeholders understand the possible
              impact.
            </p>
          </div>

          <div className="about-card process-card">
            <div className="process-number">03</div>
            <h3>Is something unusual happening?</h3>
            <p>
              Anomaly detection identifies unusual patterns or behaviour in
              project data that may require further investigation.
            </p>
          </div>

        </div>

        <p className="process-summary">
          The system then combines these insights into a comprehensive
          project risk analysis and overall risk score.
        </p>

        {/* Built for Real Infrastructure Data */}
        <div
          className="about-card bg-card"
          style={{
            backgroundImage:
              "url('https://images.unsplash.com/photo-1708357048212-1cdb297bb424?auto=format&fit=crop&w=1600&q=70')",
          }}
        >
          <div className="bg-card-overlay">
            <span className="section-label section-label-light">
              BUILT FOR REAL INFRASTRUCTURE DATA
            </span>

            <h2>Designed Around the Common Upload Form (CUF)</h2>

            <p>
              NIRMAAN is designed around the Common Upload Form (CUF) fields
              used in infrastructure project monitoring, and can analyse
              project information through multiple input methods.
            </p>

            <div className="input-methods">
              <div className="input-method-pill">
                <strong>Single Project</strong>
                <span>Analyse an individual project</span>
              </div>
              <div className="input-method-pill">
                <strong>Batch / Project History</strong>
                <span>Analyse multiple projects or historical records</span>
              </div>
              <div className="input-method-pill">
                <strong>CSV Upload</strong>
                <span>Analyse larger datasets</span>
              </div>
            </div>

            <p className="bg-card-footnote">
              This makes the platform flexible for both individual project
              assessment and portfolio-level analysis.
            </p>
          </div>
        </div>

        {/* Transparency for Citizens */}
        <div
          className="about-card bg-card"
          style={{
            backgroundImage:
              "url('https://images.unsplash.com/photo-1708064235939-0b78938aa224?auto=format&fit=crop&w=1600&q=70')",
          }}
        >
          <div className="bg-card-overlay">
            <span className="section-label section-label-light">
              TRANSPARENCY FOR CITIZENS
            </span>

            <h2>Not Just for Government Officials</h2>

            <p>
              By making project information easier to understand and
              visualize, NIRMAAN can also support greater transparency and
              public accountability. Citizens can gain visibility into:
            </p>

            <div className="citizen-tags">
              <span>Project status</span>
              <span>Project cost</span>
              <span>Expenditure</span>
              <span>Timelines</span>
              <span>Progress</span>
              <span>Implementation information</span>
            </div>

            <p className="bg-card-footnote">
              This creates a more transparent connection between public
              infrastructure projects and the people they serve.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="about-footer">
          <div className="footer-brand">
            <div className="footer-logo">N</div>
            <div>
              <strong>NIRMAAN</strong>
              <span>Project Monitoring</span>
            </div>
          </div>

          <span className="footer-text">
            Technology • Transparency • Progress
          </span>
        </div>

      </section>
      </div>
    </>
  );
}

export default About;