import { useMemo, useState } from "react";
import data from "../../data/projects-summary.json";
import { formatCr, formatCount } from "../../utils/format";
import "./StateWiseSection.css";

const COLLAPSED_COUNT = 8;

function StateWiseSection() {
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [openState, setOpenState] = useState(null);

  const maxCount = useMemo(
    () => Math.max(...data.states.map((s) => s.count)),
    []
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return data.states;
    return data.states.filter((s) => s.state.toLowerCase().includes(q));
  }, [query]);

  const isSearching = query.trim().length > 0;
  const visible = isSearching || expanded ? filtered : filtered.slice(0, COLLAPSED_COUNT);

  function toggleState(stateName) {
    setOpenState((current) => (current === stateName ? null : stateName));
  }

  return (
    <section className="nirmaan-section state-section">
      <div className="nirmaan-section-head">
        <div>
          <h2>State-wise Projects</h2>
          <p className="nirmaan-subtitle">
            {data.totals.states} states &amp; UTs, plus interstate projects,
            covering {formatCount(data.totals.projects)} tracked projects.
          </p>
        </div>

        <label className="state-search">
          <span aria-hidden="true">🔍</span>
          <input
            type="text"
            placeholder="Search state or UT..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search state or union territory"
          />
        </label>
      </div>

      {visible.length === 0 ? (
        <p className="state-empty">No state or UT matches "{query}".</p>
      ) : (
        <ul className="state-list">
          {visible.map((s) => {
            const isOpen = openState === s.state;
            const rank = data.states.findIndex((x) => x.state === s.state) + 1;
            return (
              <li key={s.state} className={`state-row ${isOpen ? "is-open" : ""}`}>
                <button
                  className="state-row-main"
                  onClick={() => toggleState(s.state)}
                  aria-expanded={isOpen}
                >
                  <span className="state-rank">{rank}</span>

                  <span className="state-name">
                    {toTitleCase(s.state)}
                  </span>

                  <span className="state-bar-track" aria-hidden="true">
                    <span
                      className="state-bar-fill"
                      style={{ width: `${(s.count / maxCount) * 100}%` }}
                    />
                  </span>

                  <span className="state-count">
                    {formatCount(s.count)} <small>projects</small>
                  </span>

                  <span className="state-cost">{formatCr(s.originalCost)}</span>

                  <span className="state-chevron" aria-hidden="true">
                    {isOpen ? "\u2212" : "+"}
                  </span>
                </button>

                {isOpen && (
                  <div className="state-detail">
                    <div className="state-detail-stats">
                      <div>
                        <span>Original (Revised) Cost</span>
                        <strong>{formatCr(s.originalCost)}</strong>
                      </div>
                      <div>
                        <span>Cumulative Expenditure</span>
                        <strong>{formatCr(s.expenditure)}</strong>
                      </div>
                    </div>

                    <p className="state-detail-label">Top projects</p>
                    <ul className="state-project-chips">
                      {(data.byState[s.state] || []).slice(0, 6).map((p) => (
                        <li key={p.code} title={p.name}>
                          <span className="chip-name">{truncate(p.name, 46)}</span>
                          <span className="chip-cost">{formatCr(p.cost)}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {!isSearching && filtered.length > COLLAPSED_COUNT && (
        <button className="nirmaan-view-all" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Show fewer states" : `View all ${filtered.length} states & UTs`}
        </button>
      )}
    </section>
  );
}

function toTitleCase(s) {
  return s.replace(/\w\S*/g, (w) => w.charAt(0) + w.slice(1).toLowerCase());
}

function truncate(s, n) {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

export default StateWiseSection;
