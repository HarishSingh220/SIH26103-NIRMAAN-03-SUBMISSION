import { useMemo } from "react";
import data from "../../data/projects-summary.json";
import { formatCr } from "../../utils/format";
import { getTopHighValueProjects, ministryFor } from "../../utils/ministry";
import "./HighValueSection.css";

function HighValueSection() {
  const topProjects = useMemo(() => getTopHighValueProjects(data, 10), []);

  const renderCards = (items, duplicate = false) => (
    <div className="hv-group" aria-hidden={duplicate ? "true" : undefined}>
      {items.map((p, i) => (
        <article key={`${duplicate ? "duplicate-" : ""}${p.code}`} className="hv-card">
          <div className="hv-card-top">
            <span className="hv-card-rank">#{i + 1}</span>
            {p.isMega && <span className="hv-mega-badge">Mega</span>}
          </div>

          <h3 className="hv-card-name" title={p.name}>
            {toTitleCase(p.name)}
          </h3>

          <p className="hv-card-meta">
            {ministryFor(p.sector)} &middot; {toTitleCase(p.state)}
          </p>

          <div className="hv-card-costs">
            <div>
              <span>Anticipated Cost</span>
              <strong>{formatCr(p.cost)}</strong>
            </div>
            <div>
              <span>Expenditure</span>
              <strong>{formatCr(p.expenditure)}</strong>
            </div>
          </div>

          {p.progress !== null && p.progress !== undefined && (
            <div
              className="hv-card-progress"
              role={duplicate ? undefined : "progressbar"}
              aria-valuemin={duplicate ? undefined : 0}
              aria-valuemax={duplicate ? undefined : 100}
              aria-valuenow={duplicate ? undefined : p.progress}
              aria-label={duplicate ? undefined : `${p.progress}% utilised`}
            >
              <span
                className="hv-card-progress-fill"
                style={{ width: `${Math.min(p.progress, 100)}%` }}
              />
            </div>
          )}

          <div className="hv-card-foot">
            <span className="hv-card-cost">
              {p.progress !== null && p.progress !== undefined
                ? `${p.progress}% utilised`
                : "—"}
            </span>
          </div>
        </article>
      ))}
    </div>
  );

  return (
    <section className="nirmaan-section hv-section">
      <div className="nirmaan-section-head">
        <div>
          <h2>💰 Top 10 High-Value Projects</h2>
          <p className="nirmaan-subtitle">
            Ranked by current anticipated cost, highest first.
          </p>
        </div>
      </div>

      <div className="hv-viewport" aria-label="Top 10 high-value projects carousel">
        <div className="hv-track">
          {renderCards(topProjects)}
          {renderCards(topProjects, true)}
        </div>
      </div>
    </section>
  );
}

function toTitleCase(s) {
  return s.replace(/\w\S*/g, (w) => w.charAt(0) + w.slice(1).toLowerCase());
}

export default HighValueSection;
