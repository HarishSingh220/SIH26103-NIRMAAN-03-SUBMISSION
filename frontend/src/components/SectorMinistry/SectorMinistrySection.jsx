import { useMemo } from "react";
import data from "../../data/projects-summary.json";
import { formatCount } from "../../utils/format";
import { getSectorBreakdown, getMinistryBreakdown } from "../../utils/ministry";
import "./SectorMinistrySection.css";

function SectorMinistrySection() {
  const sectors = useMemo(() => getSectorBreakdown(data), []);
  const ministries = useMemo(() => getMinistryBreakdown(data), []);

  const trackedCount = useMemo(
    () => sectors.reduce((sum, s) => sum + s.count, 0),
    [sectors]
  );

  const sectorMax = sectors[0]?.count || 1;
  const ministryMax = ministries[0]?.count || 1;

  return (
    <div className="sm-grid">
      <section className="nirmaan-section sm-panel">
        <div className="nirmaan-section-head">
          <div>
            <h2>📊 Sector-wise Projects</h2>
            <p className="nirmaan-subtitle">
              {formatCount(trackedCount)} tracked projects across{" "}
              {sectors.length} sectors.
            </p>
          </div>
        </div>

        <ul className="sm-list">
          {sectors.map((s) => (
            <li key={s.sector} className="sm-row">
              <span className="sm-name">{s.label}</span>
              <span className="sm-bar-track" aria-hidden="true">
                <span
                  className="sm-bar-fill sm-bar-fill--sector"
                  style={{ width: `${(s.count / sectorMax) * 100}%` }}
                />
              </span>
              <span className="sm-count">
                {formatCount(s.count)} <small>Projects</small>
              </span>
            </li>
          ))}
        </ul>
      </section>

      <section className="nirmaan-section sm-panel">
        <div className="nirmaan-section-head">
          <div>
            <h2>🏛️ Ministry-wise Projects</h2>
            <p className="nirmaan-subtitle">
              {formatCount(trackedCount)} tracked projects across{" "}
              {ministries.length} ministries/departments.
            </p>
          </div>
        </div>

        <ul className="sm-list">
          {ministries.map((m) => (
            <li key={m.ministry} className="sm-row">
              <span className="sm-name">{m.ministry}</span>
              <span className="sm-bar-track" aria-hidden="true">
                <span
                  className="sm-bar-fill sm-bar-fill--ministry"
                  style={{ width: `${(m.count / ministryMax) * 100}%` }}
                />
              </span>
              <span className="sm-count">
                {formatCount(m.count)} <small>Projects</small>
              </span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

export default SectorMinistrySection;
