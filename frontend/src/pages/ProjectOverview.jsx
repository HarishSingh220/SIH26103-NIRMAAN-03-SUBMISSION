import { useEffect, useId, useMemo, useState } from "react";
import Navbar from "../components/Navbar";
import { SearchIcon, ProjectsIcon } from "../components/Icons";
import "./ProjectOverview.css";

const DATA_URL = "/data/paiman_projects_landmark_dataset.csv";
const FIELD_LABELS = {
  project_code: "Project Code", project_name: "Project Name", sector: "Sector", state: "State / UT",
  agency_name: "Agency Name", financial_year: "Financial Year", quarter: "Quarter",
  reporting_period_date: "Reporting Period", landmark_index: "Landmark Index", n_landmarks_total: "Total Landmarks",
  horizon_months: "Horizon (months)", original_cost_rs_cr: "Original Cost (₹ Cr)",
  cumulative_expenditure_rs_cr: "Cumulative Expenditure (₹ Cr)", expenditure_to_cost_pct: "Expenditure / Cost (%)",
  project_age_at_report_months: "Project Age at Report (months)", planned_duration_months: "Planned Duration (months)",
  progress_ratio: "Progress Ratio", anticipated_cost_rs_cr: "Anticipated Cost (₹ Cr)",
  anticipated_commissioning_date: "Anticipated Commissioning Date",
  contemporaneous_cost_overrun_pct: "Contemporaneous Cost Overrun (%)",
  contemporaneous_time_overrun_pct: "Contemporaneous Time Overrun (%)",
  anticipated_cost_change_pct: "Anticipated Cost Change (%)", anticipated_date_change_months: "Anticipated Date Change (months)",
  final_cost_overrun_flag: "Final Cost Overrun Flag", final_cost_overrun_pct: "Final Cost Overrun (%)",
  final_time_overrun_flag: "Final Time Overrun Flag", final_time_overrun_pct: "Final Time Overrun (%)",
};

const HIDDEN_DETAIL_FIELDS = new Set([
  "horizon_months",
  "contemporaneous_cost_overrun_pct",
  "contemporaneous_time_overrun_pct",
  "anticipated_cost_change_pct",
  "anticipated_date_change_months",
  "final_cost_overrun_flag",
  "final_cost_overrun_pct",
  "final_time_overrun_pct",
  "final_time_overrun_flag",
]);
const FIELD_ORDER = Object.keys(FIELD_LABELS).filter((key) => !HIDDEN_DETAIL_FIELDS.has(key));

function parseCsv(text) {
  const rows = [];
  let row = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') { cell += '"'; i += 1; }
    else if (char === '"') quoted = !quoted;
    else if (char === "," && !quoted) { row.push(cell.trim()); cell = ""; }
    else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell.trim());
      if (row.some(Boolean)) rows.push(row);
      row = []; cell = "";
    } else cell += char;
  }
  if (cell || row.length) { row.push(cell.trim()); if (row.some(Boolean)) rows.push(row); }
  if (!rows.length) return [];
  const headers = rows[0];
  return rows.slice(1).map((values) => Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""])));
}

function number(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function finiteValues(records, key) {
  return records.map((r) => number(r[key])).filter((v) => v !== null);
}

function sum(records, key) { return finiteValues(records, key).reduce((a, b) => a + b, 0); }
function median(values) {
  const v = [...values].filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}
function uniqueCount(records, key) { return new Set(records.map((r) => r[key]).filter(Boolean)).size; }
function groupRecords(records, key) {
  const map = new Map();
  records.forEach((r) => { const value = r[key] || "Not available"; if (!map.has(value)) map.set(value, []); map.get(value).push(r); });
  return [...map.entries()];
}
function formatNumber(value, digits = 0) {
  if (!Number.isFinite(value)) return "Data not available";
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value);
}
function formatCr(value) {
  if (!Number.isFinite(value)) return "Data not available";
  const abs = Math.abs(value);
  if (abs >= 100000) return `₹${formatNumber(value / 100000, 2)} lakh Cr`;
  if (abs >= 1000) return `₹${formatNumber(value / 1000, 2)}k Cr`;
  return `₹${formatNumber(value, 0)} Cr`;
}
function formatPct(value, digits = 1) { return Number.isFinite(value) ? `${formatNumber(value, digits)}%` : "Data not available"; }
function labelFor(value) { return value?.length > 26 ? `${value.slice(0, 25)}…` : value; }
function parseDate(value) { const d = new Date(value); return Number.isNaN(d.getTime()) ? null : d; }
function shortDate(value) { const d = parseDate(value); return d ? d.toLocaleDateString("en-IN", { month: "short", year: "numeric" }) : "Data not available"; }
function projectLatest(records) {
  return [...records].sort((a, b) => String(b.reporting_period_date).localeCompare(String(a.reporting_period_date)))[0];
}
function latestPerProject(records) {
  const map = new Map();
  records.forEach((r) => {
    const current = map.get(r.project_code);
    if (!current || String(r.reporting_period_date) > String(current.reporting_period_date)) map.set(r.project_code, r);
  });
  return [...map.values()];
}
function flagLabel(value, type) {
  if (value === "1" || value === 1) return type === "cost" ? "Overrun" : "Overrun";
  if (value === "0" || value === 0) return "No overrun";
  return "Data not available";
}

function TricolorLine() { return <span className="overview-tricolor"><i /><i /><i /></span>; }

function EmptyState({ message = "Data not available" }) {
  return <div className="overview-empty">{message}</div>;
}

function BarChart({ items, valueFormatter = formatNumber, emptyMessage = "Data not available" }) {
  const max = Math.max(...items.map((i) => i.value || 0), 0);
  if (!items.length || max <= 0) return <EmptyState message={emptyMessage} />;
  return <div className="bar-chart">
    {items.map((item) => <div className="bar-item" key={item.label}>
      <div className="bar-label" title={item.label}>{labelFor(item.label)}</div>
      <div className="bar-track"><span style={{ width: `${Math.max(2, (item.value / max) * 100)}%` }} /></div>
      <strong>{valueFormatter(item.value)}</strong>
    </div>)}
  </div>;
}

function DonutChart({ items, centerLabel = "Records" }) {
  const total = items.reduce((a, b) => a + b.value, 0);
  if (!total) return <EmptyState />;
  const colors = ["#2f63a3", "#e98222", "#2b8a57", "#7d8fa2"];
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const segments = items.reduce((acc, item) => {
    const prevCumulative = acc.length ? acc[acc.length - 1].cumulative : 0;
    acc.push({ ...item, cumulative: prevCumulative + item.value / total });
    return acc;
  }, []);
  return <div className="donut-wrap">
    <div className="donut-chart">
      <svg viewBox="0 0 140 140" aria-label="Distribution chart">
        <circle cx="70" cy="70" r={radius} fill="none" stroke="#edf1f4" strokeWidth="20" />
        {segments.map((item, i) => {
          const dash = (item.value / total) * circumference;
          const prevCumulative = i === 0 ? 0 : segments[i - 1].cumulative;
          const offset = -prevCumulative * circumference;
          return <circle key={item.label} cx="70" cy="70" r={radius} fill="none" stroke={colors[i % colors.length]} strokeWidth="20" strokeDasharray={`${dash} ${circumference - dash}`} strokeDashoffset={offset} transform="rotate(-90 70 70)" />;
        })}
        <text x="70" y="67" textAnchor="middle" className="donut-total">{formatNumber(total)}</text>
        <text x="70" y="83" textAnchor="middle" className="donut-caption">{centerLabel}</text>
      </svg>
    </div>
    <div className="legend-list">{items.map((item, i) => <div key={item.label} className="legend-item"><span style={{ background: colors[i % colors.length] }} /> <span>{item.label}</span><strong>{formatNumber(item.value)}</strong></div>)}</div>
  </div>;
}

function LineChart({ points, yFormatter = formatNumber, emptyMessage = "Data not available", large = false, compact = false }) {
  const valid = points.filter((p) => Number.isFinite(p.value));
  const clipId = `line-plot-clip-${useId().replace(/:/g, "")}`;
  if (valid.length < 2) return <EmptyState message={emptyMessage} />;

  const width = large ? 920 : compact ? 720 : 720;
  const height = large ? 320 : compact ? 230 : 250;
  const pad = large
    ? { left: 78, right: 30, top: 24, bottom: 68 }
    : compact
      ? { left: 82, right: 38, top: 20, bottom: 56 }
      : { left: 72, right: 24, top: 20, bottom: 58 };
  const min = Math.min(...valid.map((p) => p.value));
  const max = Math.max(...valid.map((p) => p.value));
  const range = max - min || 1;
  const chartWidth = width - pad.left - pad.right;
  const horizontalInset = compact ? Math.min(20, chartWidth / Math.max(valid.length * 4, 1)) : 0;
  const x = (i) => pad.left + horizontalInset + (i / (valid.length - 1)) * (chartWidth - horizontalInset * 2);
  const y = (v) => pad.top + (1 - (v - min) / range) * (height - pad.top - pad.bottom);
  const path = valid.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.value)}`).join(" ");
  const area = `${path} L ${x(valid.length - 1)},${height - pad.bottom} L ${x(0)},${height - pad.bottom} Z`;
  return <div className={`line-chart${compact ? " line-chart--compact" : ""}`}>
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" role="img">
      <defs>
        <clipPath id={clipId}>
          <rect x={pad.left} y={pad.top} width={chartWidth} height={height - pad.top - pad.bottom} />
        </clipPath>
      </defs>
      {[0, .5, 1].map((t) => <g key={t}>
        <line x1={pad.left} x2={width - pad.right} y1={pad.top + t * (height - pad.top - pad.bottom)} y2={pad.top + t * (height - pad.top - pad.bottom)} className="grid-line" />
        <text x={pad.left - 8} y={pad.top + t * (height - pad.top - pad.bottom) + 4} textAnchor="end" className="axis-text">{yFormatter(max - t * range)}</text>
      </g>)}
      <g clipPath={`url(#${clipId})`}>
        <path d={area} className="area-fill" />
        <path d={path} className="line-path" />
        {valid.map((p, i) => <circle key={`${p.label}-${i}`} cx={x(i)} cy={y(p.value)} r="3.5" className="line-dot"><title>{`${p.label}: ${yFormatter(p.value)}`}</title></circle>)}
      </g>
      {valid.map((p, i) => {
        const crowded = valid.length > 6;
        const compactLabels = compact && crowded;
        const maxLabels = large ? 12 : compact ? 7 : 9;
        const labelStep = crowded ? Math.max(1, Math.ceil(valid.length / maxLabels)) : 1;
        const isLast = i === valid.length - 1;
        if (crowded && !compactLabels && i % labelStep !== 0 && !isLast) return null;
        const rotate = crowded && !compactLabels;
        const labelY = compactLabels ? height - 18 : crowded ? height - 10 : height - 18;
        return <text
          key={`x-${p.label}-${i}`}
          x={x(i)}
          y={labelY}
          textAnchor={rotate ? "end" : "middle"}
          transform={rotate ? `rotate(-40 ${x(i)} ${labelY})` : undefined}
          className={`axis-text${compactLabels ? " axis-text-compact" : ""}`}
        >{labelFor(p.label)}</text>;
      })}
    </svg>
  </div>;
}

function CostExpenditureChart({ costPoints, expenditurePoints, emptyMessage = "Data not available" }) {
  const byYear = new Map();
  costPoints.forEach((p) => byYear.set(p.label, { label: p.label, cost: p.value, expenditure: 0 }));
  expenditurePoints.forEach((p) => {
    const current = byYear.get(p.label) || { label: p.label, cost: 0, expenditure: 0 };
    current.expenditure = p.value;
    byYear.set(p.label, current);
  });
  const points = [...byYear.values()].filter((p) => Number.isFinite(p.cost) || Number.isFinite(p.expenditure));
  if (points.length < 1) return <EmptyState message={emptyMessage} />;

  const width = 920, height = 320, pad = { left: 78, right: 30, top: 28, bottom: 64 };
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const maxValue = Math.max(...points.flatMap((p) => [p.cost || 0, p.expenditure || 0]), 1);
  const step = chartWidth / points.length;
  const barWidth = Math.min(34, step * 0.48);
  const x = (i) => pad.left + step * i + step / 2;
  const y = (v) => pad.top + (1 - (v / maxValue)) * chartHeight;
  const linePath = points.map((p, i) => `${i ? "L" : "M"} ${x(i)} ${y(p.cost || 0)}`).join(" ");

  return <div className="cost-expenditure-chart">
    <div className="chart-legend" aria-label="Chart legend">
      <span><i className="legend-swatch legend-swatch-bar" /> Expenditure</span>
      <span><i className="legend-swatch legend-swatch-line" /> Cost</span>
    </div>
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="Cost and expenditure by financial year">
      {[0, .25, .5, .75, 1].map((t) => {
        const value = maxValue * (1 - t);
        const yPos = pad.top + t * chartHeight;
        return <g key={t}>
          <line x1={pad.left} x2={width - pad.right} y1={yPos} y2={yPos} className="grid-line" />
          <text x={pad.left - 9} y={yPos + 4} textAnchor="end" className="axis-text">{formatCr(value)}</text>
        </g>;
      })}
      {points.map((p, i) => {
        const value = p.expenditure || 0;
        const barHeight = chartHeight - (y(value) - pad.top);
        return <rect key={`bar-${p.label}`} x={x(i) - barWidth / 2} y={y(value)} width={barWidth} height={Math.max(0, barHeight)} rx="3" className="expenditure-bar">
          <title>{`${p.label}: Expenditure ${formatCr(value)}`}</title>
        </rect>;
      })}
      <path d={linePath} className="cost-line" />
      {points.map((p, i) => <circle key={`dot-${p.label}`} cx={x(i)} cy={y(p.cost || 0)} r="4" className="cost-dot">
        <title>{`${p.label}: Cost ${formatCr(p.cost)}`}</title>
      </circle>)}
      {points.map((p, i) => {
        const crowded = points.length > 6;
        return <text key={`x-${p.label}`} x={x(i)} y={crowded ? height - 9 : height - 15} textAnchor={crowded ? "end" : "middle"} transform={crowded ? `rotate(-32 ${x(i)} ${height - 9})` : undefined} className="axis-text">{labelFor(p.label)}</text>;
      })}
    </svg>
  </div>;
}
function ChartPanel({ title, subtitle, children, wide = false }) {
  return <section className={`overview-panel ${wide ? "overview-panel-wide" : ""}`}>
    <div className="panel-heading"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div></div>
    {children}
  </section>;
}

function ProjectOverview() {
  const [records, setRecords] = useState([]);
  const [status, setStatus] = useState({ loading: true, error: "" });
  const [query, setQuery] = useState("");
  const [selectedCode, setSelectedCode] = useState("");
  const [scope, setScope] = useState("all");

  useEffect(() => {
    let cancelled = false;
    fetch(DATA_URL)
      .then((response) => { if (!response.ok) throw new Error(`CSV request failed (${response.status})`); return response.text(); })
      .then((text) => { if (!cancelled) { setRecords(parseCsv(text)); setStatus({ loading: false, error: "" }); } })
      .catch((error) => { if (!cancelled) setStatus({ loading: false, error: error.message }); });
    return () => { cancelled = true; };
  }, []);

  const projectGroups = useMemo(() => {
    const map = new Map();
    records.forEach((r) => { if (!map.has(r.project_code)) map.set(r.project_code, []); map.get(r.project_code).push(r); });
    return map;
  }, [records]);

  const searchResults = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return [...projectGroups.entries()].map(([code, rows]) => ({ code, row: projectLatest(rows) }))
      .filter(({ code, row }) => [code, row.project_name, row.agency_name, row.sector, row.state].some((v) => String(v || "").toLowerCase().includes(q)))
      .slice(0, 12);
  }, [query, projectGroups]);

  const selectedRecords = useMemo(() => selectedCode ? projectGroups.get(selectedCode) || [] : [], [selectedCode, projectGroups]);
  const selectedLatest = selectedRecords.length ? projectLatest(selectedRecords) : null;

  const years = useMemo(() => [...new Set(records.map((r) => r.financial_year).filter(Boolean))].sort(), [records]);
  const scopedRecords = useMemo(() => scope === "all" ? records : records.filter((r) => r.financial_year === scope), [records, scope]);
  const uniqueProjects = useMemo(() => new Set(scopedRecords.map((r) => r.project_code)).size, [scopedRecords]);
  const scopedLatest = useMemo(() => latestPerProject(scopedRecords), [scopedRecords]);

  const sectorItems = useMemo(() => groupRecords(scopedRecords, "sector").map(([label, rows]) => ({ label, value: new Set(rows.map(r => r.project_code)).size })).sort((a,b)=>b.value-a.value).slice(0,8), [scopedRecords]);
  const stateItems = useMemo(() => groupRecords(scopedRecords, "state").map(([label, rows]) => ({ label, value: new Set(rows.map(r => r.project_code)).size })).sort((a,b)=>b.value-a.value).slice(0,10), [scopedRecords]);
  const agencyItems = useMemo(() => groupRecords(scopedRecords, "agency_name").map(([label, rows]) => ({ label, value: new Set(rows.map(r => r.project_code)).size })).sort((a,b)=>b.value-a.value).slice(0,8), [scopedRecords]);
  const yearItems = useMemo(() => years.map((year) => ({ label: year, value: new Set(records.filter(r=>r.financial_year===year).map(r=>r.project_code)).size })), [records, years]);
  const quarterItems = useMemo(() => ["Q1","Q2","Q3","Q4"].map((q)=>({label:q,value:scopedRecords.filter(r=>r.quarter===q).length})), [scopedRecords]);
  const costItems = useMemo(() => groupRecords(scopedLatest, "sector").map(([label, rows]) => ({ label, value: sum(rows, "original_cost_rs_cr") })).sort((a,b)=>b.value-a.value).slice(0,7), [scopedLatest]);
  const expenditureItems = useMemo(() => groupRecords(scopedLatest, "sector").map(([label, rows]) => ({ label, value: sum(rows, "cumulative_expenditure_rs_cr") })).sort((a,b)=>b.value-a.value).slice(0,7), [scopedLatest]);
  const progressItems = useMemo(() => groupRecords(scopedLatest, "sector").map(([label, rows]) => ({ label, value: median(finiteValues(rows, "progress_ratio")) })).filter(i=>i.value!==null).sort((a,b)=>b.value-a.value).slice(0,7), [scopedLatest]);
  const annualExpenditure = useMemo(() => years.map(year=>({label:year,value:sum(latestPerProject(records.filter(r=>r.financial_year===year)),"cumulative_expenditure_rs_cr")})).filter(i=>i.value>0), [records, years]);
  const annualCost = useMemo(() => years.map(year=>({label:year,value:sum(latestPerProject(records.filter(r=>r.financial_year===year)),"anticipated_cost_rs_cr")})).filter(i=>i.value>0), [records, years]);
  const search = (event) => {
    event.preventDefault();
    if (searchResults[0]) { setSelectedCode(searchResults[0].code); setQuery(searchResults[0].code); }
  };

  return <>
    <Navbar />
    <main className="overview-page">
      <header className="overview-header">
        <div>
          <TricolorLine />
          <p className="overview-eyebrow">IPMD • PROJECT OVERVIEW</p>
          <h1>Infrastructure Project Intelligence</h1>
          <p className="overview-subtitle">A data-led view of project scale, expenditure, progress and monitoring history, built only from the supplied PAIMANA project records.</p>
        </div>
        <div className="overview-source"><span>Source</span><strong>PAIMANA CSV</strong><small>{status.loading ? "Loading records…" : status.error ? "Data not available" : `${formatNumber(records.length)} reporting records`}</small></div>
      </header>

      {status.loading ? <div className="overview-loading"><ProjectsIcon /><strong>Loading project records</strong><span>Reading the supplied CSV…</span></div> : status.error ? <div className="overview-error"><strong>Project data could not be loaded.</strong><span>{status.error}</span></div> : <>
        <div className="overview-toolbar">
          <div><span className="toolbar-label">Dashboard scope</span><div className="scope-tabs"><button className={scope==="all"?"active":""} onClick={()=>setScope("all")} type="button">All years</button>{years.map(y=><button key={y} className={scope===y?"active":""} onClick={()=>setScope(y)} type="button">{y}</button>)}</div></div>
          <div className="toolbar-meta"><strong>{formatNumber(uniqueProjects)}</strong><span>unique projects in view</span><i /> <strong>{formatNumber(scopedRecords.length)}</strong><span>reporting records</span></div>
        </div>

        <section className="overview-kpis">
          <div className="overview-kpi"><span>Unique projects</span><strong>{formatNumber(uniqueProjects)}</strong><small>{formatNumber(scopedRecords.length)} historical records in view</small></div>
          <div className="overview-kpi saffron"><span>Original project cost</span><strong>{formatCr(sum(scopedLatest,"original_cost_rs_cr"))}</strong><small>Latest record per project in view</small></div>
          <div className="overview-kpi green"><span>Cumulative expenditure</span><strong>{formatCr(sum(scopedLatest,"cumulative_expenditure_rs_cr"))}</strong><small>Latest reported value per project</small></div>
          <div className="overview-kpi navy"><span>States / territories</span><strong>{formatNumber(uniqueCount(scopedRecords,"state"))}</strong><small>Distinct values present in CSV</small></div>
        </section>

        <div className="overview-analytics">
          <section className="overview-section-group">
            <div className="overview-section-heading">
              <h2>Project Distribution &amp; Analysis</h2>
              <p>Project distribution across sectors, states / territories and agencies.</p>
            </div>
            <div className="overview-grid overview-grid-distribution">
              <ChartPanel title="Sector-wise project analysis" subtitle="Unique project codes represented by sector."><BarChart items={sectorItems} /></ChartPanel>
              <ChartPanel title="State-wise project analysis" subtitle="Top states / territories by unique project count."><BarChart items={stateItems} /></ChartPanel>
              <ChartPanel title="Agency-wise project analysis" subtitle="Ministry is not a CSV field; agency is the available administrative dimension."><BarChart items={agencyItems} /></ChartPanel>
            </div>
          </section>

          <section className="overview-section-group">
            <div className="overview-section-heading">
              <h2>Reporting &amp; Financial Year Trends</h2>
              <p>Reporting activity and project distribution across financial years.</p>
            </div>
            <div className="overview-grid overview-grid-two">
              <ChartPanel title="Quarter reporting pattern" subtitle="Number of reporting records by quarter."><DonutChart items={quarterItems} centerLabel="Records" /></ChartPanel>
              <ChartPanel title="Projects by financial year" subtitle="Unique project codes appearing in each financial year."><BarChart items={yearItems} /></ChartPanel>
            </div>
          </section>

          <section className="overview-section-group">
            <div className="overview-section-heading">
              <h2>Cost, Expenditure &amp; Progress Analysis</h2>
              <p>Financial scale, expenditure and progress indicators from the supplied project records.</p>
            </div>
            <div className="overview-grid">
              <ChartPanel title="Original cost by sector" subtitle="Sum of original cost using the latest available record per project."><BarChart items={costItems} valueFormatter={formatCr} /></ChartPanel>
              <ChartPanel title="Expenditure by sector" subtitle="Latest reported cumulative expenditure per project."><BarChart items={expenditureItems} valueFormatter={formatCr} /></ChartPanel>
              <ChartPanel title="Median progress ratio by sector" subtitle="Only finite values in the progress_ratio field are plotted."><BarChart items={progressItems} valueFormatter={(v)=>formatNumber(v,2)} /></ChartPanel>
              <ChartPanel title="Annual cumulative expenditure" subtitle="Latest available project snapshot within each financial year."><LineChart points={annualExpenditure} yFormatter={formatCr} compact /></ChartPanel>
              <ChartPanel title="Annual anticipated cost" subtitle="Latest available project snapshot within each financial year."><LineChart points={annualCost} yFormatter={formatCr} compact /></ChartPanel>
              <ChartPanel title="Cost &amp; expenditure" subtitle="Annual latest-project snapshots from the supplied CSV. Expenditure is shown as bars and anticipated cost as a line." wide><CostExpenditureChart costPoints={annualCost} expenditurePoints={annualExpenditure} /></ChartPanel>
            </div>
          </section>
        </div>

        <section className="selected-project-section">
          <div className="selected-heading"><div><TricolorLine /><p className="overview-eyebrow">PROJECT DETAIL & HISTORY</p><h2>{selectedLatest ? selectedLatest.project_name : "Select a project to inspect its history"}</h2><p>{selectedLatest ? `${selectedLatest.project_code} • ${selectedLatest.agency_name || "Agency not reported"}` : "Search for a project below to open all available records for one project."}</p></div>{selectedLatest && <button type="button" className="btn btn-secondary" onClick={()=>setSelectedCode("")}>Clear selection</button>}</div>
          <form className="project-search project-search--details" onSubmit={search}>
            <SearchIcon />
            <input value={query} onChange={(e)=>setQuery(e.target.value)} placeholder="Search by Project Code, Project Name, Agency, Sector or State" aria-label="Search projects" />
            <button className="btn btn-primary" type="submit" disabled={!searchResults.length}>Find project</button>
            {query && searchResults.length > 0 && <div className="search-results">{searchResults.map(({code,row})=><button key={code} type="button" onClick={()=>{setSelectedCode(code);setQuery(row.project_name);}}><strong>{code}</strong><span>{row.project_name}</span><small>{row.sector} • {row.state}</small></button>)}</div>}
          </form>
          {!selectedLatest ? <div className="project-detail-empty">Search for a project code or name to reveal its complete CSV-backed profile and historical quarterly records.</div> : <div className="project-detail-layout">
            <div className="project-profile">
              <div className="profile-top"><span>Latest reporting record</span><strong>{shortDate(selectedLatest.reporting_period_date)}</strong></div>
              <div className="profile-grid">{FIELD_ORDER.map((key)=><div className="profile-field" key={key}><span>{FIELD_LABELS[key]}</span><strong>{key.includes("cost_rs_cr") ? formatCr(number(selectedLatest[key])) : key.endsWith("_pct") ? formatPct(number(selectedLatest[key])) : key==="reporting_period_date" ? shortDate(selectedLatest[key]) : key.includes("date") ? (selectedLatest[key] || "Data not available") : (selectedLatest[key] || "Data not available")}</strong></div>)}</div>
            </div>
            <div className="history-panel">
              <div className="panel-heading"><div><h2>Historical expenditure</h2><p>Quarter-by-quarter records for {selectedLatest.project_code}.</p></div><span className="record-pill">{formatNumber(selectedRecords.length)} records</span></div>
              <LineChart points={[...selectedRecords].sort((a,b)=>String(a.reporting_period_date).localeCompare(String(b.reporting_period_date))).map(r=>({label:`${r.financial_year} ${r.quarter}`,value:number(r.cumulative_expenditure_rs_cr)}))} yFormatter={formatCr} large />
              <br/>
              <div className="history-table-wrap"><table><thead><tr><th>Period</th><th>Expenditure</th><th>Exp. / cost</th><th>Progress ratio</th><th>Anticipated cost</th></tr></thead><tbody>{[...selectedRecords].sort((a,b)=>String(b.reporting_period_date).localeCompare(String(a.reporting_period_date))).map((r,i)=><tr key={`${r.reporting_period_date}-${i}`}><td>{r.financial_year} {r.quarter}</td><td>{formatCr(number(r.cumulative_expenditure_rs_cr))}</td><td>{formatPct(number(r.expenditure_to_cost_pct))}</td><td>{Number.isFinite(number(r.progress_ratio)) ? formatNumber(number(r.progress_ratio),2) : "Data not available"}</td><td>{formatCr(number(r.anticipated_cost_rs_cr))}</td></tr>)}</tbody></table></div>
            </div>
          </div>}
        </section>
      </>}
    </main>
  </>;
}

export default ProjectOverview;
