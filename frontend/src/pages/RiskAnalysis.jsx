import { useMemo, useRef, useState } from "react";
import Navbar from "../components/Navbar";
import { RiskIcon } from "../components/Icons";
import "./RiskAnalysis.css";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");
const PREFERRED_MODEL = "model_b";

const FIELD_DEFINITIONS = [
  { key: "projectCode", label: "Project Code", type: "text", required: true, group: "project" },
  { key: "projectName", label: "Project Name", type: "text", required: false, group: "project" },
  { key: "agencyName", label: "Agency Name", type: "text", required: true, group: "project" },
  { key: "sector", label: "Sector", type: "select", required: true, group: "project", options: ["ROAD TRANSPORT AND HIGHWAYS", "RAILWAYS", "PETROLEUM", "COAL", "POWER", "URBAN DEVELOPMENT", "WATER RESOURCES", "HEALTH AND FAMILY WELFARE", "CIVIL AVIATION", "STEEL", "DEPARTMENT OF HIGHER EDUCATION", "TELECOMMUNICATIONS", "MINES", "ATOMIC ENERGY", "SHIPPING AND PORTS", "DPIIT", "FERTILISERS", "HOME AFFAIRS", "HUMAN RESOURCE DEVELOPMENT", "HEAVY INDUSTRY", "DEFENCE PRODUCTION", "RURAL DEVELOPMENT", "COMMERCE", "SOCIAL JUSTICE", "DONER", "COMMERCE AND INDUSTRY", "INFORMATION AND BROADCASTING", "RENEWABLE ENERGY", "PETROCHEMICALS", "CULTURE DEVELOPMENT"] },
  { key: "state", label: "State / UT", type: "text", required: true, group: "project" },
  { key: "projectStatus", label: "Project Status", type: "select", required: false, group: "project", options: ["On Track", "Minor Delay", "Delayed", "Critical", "Completed", "Reporting"] },
  { key: "reportingQuarter", label: "Reporting Quarter", type: "select", required: true, group: "financial", options: ["Q1", "Q2", "Q3", "Q4"] },
  { key: "financialQuarter", label: "Financial Quarter", type: "select", required: false, group: "financial", options: ["Q1", "Q2", "Q3", "Q4"] },
  { key: "financialYear", label: "Financial Year", type: "text", required: true, group: "financial", placeholder: "2026-27" },
  { key: "originalCost", label: "Original Cost (₹ Cr)", type: "number", required: true, group: "financial", step: "0.01" },
  { key: "anticipatedCost", label: "Anticipated Cost (₹ Cr)", type: "number", required: true, group: "financial", step: "0.01" },
  { key: "cumulativeExpenditure", label: "Cumulative Expenditure (₹ Cr)", type: "number", required: true, group: "financial", step: "0.01" },
  { key: "approvalDate", label: "Approval Date", type: "date", required: true, group: "timeline" },
  { key: "originalCommissioningDate", label: "Original Commissioning Date", type: "date", required: true, group: "timeline" },
  { key: "anticipatedCommissioningDate", label: "Anticipated Commissioning Date", type: "date", required: true, group: "timeline" },
];

const UI_COLUMNS = FIELD_DEFINITIONS.map(({ key }) => key);
const REQUIRED_FIELD_KEYS = FIELD_DEFINITIONS.filter(({ required }) => required).map(({ key }) => key);

const BACKEND_REQUIRED_COLUMNS = [
  "project_code",
  "agency_name",
  "sector",
  "state",
  "reporting_quarter",
  "financial_year",
  "original_cost_rs_cr",
  "anticipated_cost_rs_cr",
  "cumulative_expenditure_rs_cr",
  "approval_date",
  "original_commissioning_date",
  "anticipated_commissioning_date",
];

const BACKEND_OPTIONAL_COLUMNS = [
  "project_name",
  "project_status",
  "revised_cost_rs_cr",
  "revised_commissioning_date",
];

const HEADER_ALIASES = {
  projectcode: "projectCode",
  project_code: "projectCode",
  projectname: "projectName",
  project_name: "projectName",
  agencyname: "agencyName",
  agency_name: "agencyName",
  sector: "sector",
  state: "state",
  statenames: "state",
  projectstatus: "projectStatus",
  project_status: "projectStatus",
  reportingquarter: "reportingQuarter",
  reporting_quarter: "reportingQuarter",
  financialquarter: "financialQuarter",
  financial_quarter: "financialQuarter",
  financialyear: "financialYear",
  financial_year: "financialYear",
  originalcost: "originalCost",
  original_cost_rs_cr: "originalCost",
  anticipatedcost: "anticipatedCost",
  anticipated_cost_rs_cr: "anticipatedCost",
  cumulativeexpenditure: "cumulativeExpenditure",
  cumulative_expenditure_rs_cr: "cumulativeExpenditure",
  approvaldate: "approvalDate",
  approval_date: "approvalDate",
  originalcommissioningdate: "originalCommissioningDate",
  original_commissioning_date: "originalCommissioningDate",
  anticipatedcommissioningdate: "anticipatedCommissioningDate",
  anticipated_commissioning_date: "anticipatedCommissioningDate",
};

const API_FIELD_BY_UI_KEY = {
  projectCode: "project_code",
  projectName: "project_name",
  agencyName: "agency_name",
  sector: "sector",
  state: "state",
  projectStatus: "project_status",
  reportingQuarter: "reporting_quarter",
  financialYear: "financial_year",
  originalCost: "original_cost_rs_cr",
  anticipatedCost: "anticipated_cost_rs_cr",
  cumulativeExpenditure: "cumulative_expenditure_rs_cr",
  approvalDate: "approval_date",
  originalCommissioningDate: "original_commissioning_date",
  anticipatedCommissioningDate: "anticipated_commissioning_date",
};

function createRowId() {
  return `risk-row-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function emptyRecord() {
  return {
    _rowId: createRowId(),
    ...Object.fromEntries(UI_COLUMNS.map((key) => [key, ""])),
  };
}

function displayValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

function numberValue(value, digits = 2) {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function percentValue(value) {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function riskClass(value) {
  const normalized = String(value || "").toLowerCase();
  if (normalized.includes("high") || normalized.includes("critical")) return "high";
  if (normalized.includes("medium") || normalized.includes("moderate") || normalized.includes("minor")) return "medium";
  if (normalized.includes("low") || normalized.includes("on track")) return "low";
  return "neutral";
}

function probabilityRisk(probability) {
  if (probability === null || probability === undefined || Number.isNaN(Number(probability))) return "Not returned";
  const p = Number(probability);
  if (p > 0.7) return "High";
  if (p > 0.4) return "Medium";
  return "Low";
}

function classificationReason(block, label) {
  if (!block) return `${label} classification result was not returned.`;
  if (block.reason) return block.reason;
  if (block.prediction === "Overrun Risk") return `${label} overrun risk was flagged by the classifier at ${percentValue(block.probability)} probability.`;
  return `${label} was not flagged as overrun risk (${percentValue(block.probability)} probability).`;
}

function toRawRow(record) {
  const output = {};
  Object.entries(API_FIELD_BY_UI_KEY).forEach(([uiKey, apiKey]) => {
    const value = record[uiKey];
    if (value !== "" && value !== null && value !== undefined) {
      output[apiKey] = ["original_cost_rs_cr", "anticipated_cost_rs_cr", "cumulative_expenditure_rs_cr"].includes(apiKey)
        ? Number(value)
        : value;
    }
  });
  return output;
}

function sourceMetadata(rows) {
  const map = new Map();
  rows.forEach((row) => {
    const code = String(row.projectCode || "").trim();
    if (code) map.set(code, row);
  });
  return map;
}

function normalizePrediction(item, sourceRecord, index) {
  const output = item?.prediction || item?.result || item || {};
  const classification = output.classification || {};
  const costClassification = classification.cost_overrun || {};
  const timeClassification = classification.time_overrun || {};
  const overrun = output.overrun || {};
  const anomaly = output.anomaly || {};

  const costProbability = costClassification.probability;
  const timeProbability = timeClassification.probability;
  const overallProbability = Math.max(
    Number(costProbability ?? 0),
    Number(timeProbability ?? 0),
  );

  return {
    id: `${output.project_code || sourceRecord?.projectCode || "project"}-${index}`,
    projectCode: output.project_code || sourceRecord?.projectCode,
    projectName: sourceRecord?.projectName,
    agencyName: sourceRecord?.agencyName,
    state: sourceRecord?.state,
    sector: sourceRecord?.sector,
    costRisk: probabilityRisk(costProbability),
    timeRisk: probabilityRisk(timeProbability),
    overallRisk: probabilityRisk(overallProbability),
    costProbability,
    timeProbability,
    costPrediction: costClassification.prediction,
    timePrediction: timeClassification.prediction,
    costReason: classificationReason(costClassification, "Cost"),
    timeReason: classificationReason(timeClassification, "Time"),
    overrun,
    anomaly,
    warnings: output.warnings || [],
    rowsUsed: output.rows_used,
    reason: [
      classificationReason(costClassification, "Cost"),
      classificationReason(timeClassification, "Time"),
      overrun.cost_overrun?.reason,
      overrun.time_overrun?.reason,
      anomaly.anomaly_reason,
    ].filter((value) => value),
    raw: output,
  };
}

function normalizePayload(payload, rows) {
  const metadata = sourceMetadata(rows);
  const predictions = Array.isArray(payload?.predictions)
    ? payload.predictions
    : [payload?.prediction ?? payload?.result ?? payload];

  return predictions
    .filter((prediction) => prediction && !prediction.error)
    .map((prediction, index) => normalizePrediction(
      prediction,
      metadata.get(String(prediction.project_code || "").trim()) || rows[index] || rows[0] || emptyRecord(),
      index,
    ));
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && quoted && next === '"') {
      cell += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell.trim());
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell.trim());
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  if (cell !== "" || row.length) {
    row.push(cell.trim());
    if (row.some((value) => value !== "")) rows.push(row);
  }

  return rows;
}

function RiskBadge({ value }) {
  return <span className={`risk-badge risk-badge-${riskClass(value)}`}>{displayValue(value)}</span>;
}

function Field({ definition, value, onChange }) {
  const common = {
    id: definition.key,
    name: definition.key,
    value: value ?? "",
    onChange: (event) => onChange(definition.key, event.target.value),
    required: definition.required,
  };

  return (
    <label className="risk-field" htmlFor={definition.key}>
      <span>{definition.label}{definition.required ? <em> *</em> : <small> (Optional)</small>}</span>
      {definition.type === "select" ? (
        <select {...common}>
          <option value="">Select {definition.label.toLowerCase()}</option>
          {definition.options.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
      ) : (
        <input {...common} type={definition.type} step={definition.step} placeholder={definition.placeholder} />
      )}
    </label>
  );
}

function LineChart({ series, valueFormatter = (value) => numberValue(value, 1), emptyLabel }) {
  const width = 720;
  const height = 250;
  const padding = { top: 18, right: 24, bottom: 42, left: 44 };
  const dates = [...new Set(series.flatMap((entry) => entry.values.map((point) => point.date)).filter(Boolean))].sort();

  if (dates.length === 0) {
    return <div className="chart-empty">{emptyLabel}</div>;
  }

  const indexedValues = series.flatMap((entry) => entry.values).filter((point) => Number.isFinite(Number(point.value)));
  if (indexedValues.length === 0) {
    return <div className="chart-empty">{emptyLabel}</div>;
  }

  const min = Math.min(0, ...indexedValues.map((point) => Number(point.value)));
  const max = Math.max(...indexedValues.map((point) => Number(point.value)), 1);
  const range = Math.max(max - min, 1);
  const xStep = dates.length > 1 ? (width - padding.left - padding.right) / (dates.length - 1) : 0;
  const xForDate = (date) => padding.left + Math.max(0, dates.indexOf(date)) * xStep;
  const y = (value) => height - padding.bottom - ((Number(value) - min) / range) * (height - padding.top - padding.bottom);

  return (
    <div className="line-chart-wrap">
      <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Time series risk chart">
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} className="chart-axis" />
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} className="chart-axis" />
        {[0, 0.5, 1].map((ratio) => {
          const value = min + range * ratio;
          const yPos = y(value);
          return (
            <g key={ratio}>
              <line x1={padding.left} y1={yPos} x2={width - padding.right} y2={yPos} className="chart-gridline" />
              <text x={padding.left - 8} y={yPos + 4} className="chart-label" textAnchor="end">{valueFormatter(value)}</text>
            </g>
          );
        })}

        {series.map((entry, seriesIndex) => {
          const valid = entry.values.filter((point) => Number.isFinite(Number(point.value)));
          const points = valid.map((point) => `${xForDate(point.date)},${y(point.value)}`).join(" ");
          return (
            <g key={entry.key}>
              {valid.length > 1 && <polyline points={points} fill="none" className={`series-line series-${seriesIndex}`} />}
              {valid.map((point) => (
                <circle key={`${entry.key}-${point.date}`} cx={xForDate(point.date)} cy={y(point.value)} r="4" className={`series-point series-${seriesIndex}`} />
              ))}
            </g>
          );
        })}

        {dates.map((date) => (
          <text key={date} x={xForDate(date)} y={height - padding.bottom + 25} className="chart-label" textAnchor="middle">
            {String(date).slice(0, 7)}
          </text>
        ))}
      </svg>
      <div className="chart-legend">
        {series.map((entry, index) => <span key={entry.key}><i className={`legend-dot series-${index}`} />{entry.label}</span>)}
      </div>
    </div>
  );
}

function RiskAnalysis() {
  const [mode, setMode] = useState("single");
  const [rows, setRows] = useState([emptyRecord()]);
  const [singleRecord, setSingleRecord] = useState(emptyRecord());
  const [batchRows, setBatchRows] = useState([emptyRecord()]);
  const [results, setResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [validation, setValidation] = useState([]);
  const [apiState, setApiState] = useState({ type: "idle", message: "" });
  const [csvName, setCsvName] = useState("");
  const [csvFile, setCsvFile] = useState(null);
  const [summaryState, setSummaryState] = useState({ type: "idle", text: "", model: "", warnings: [] });
  const [selectedProjectCode, setSelectedProjectCode] = useState("");
  const [analysisProjectCode, setAnalysisProjectCode] = useState("");
  const fileRef = useRef(null);

  const canAnalyze = rows.length > 0 && rows.every((row) => REQUIRED_FIELD_KEYS.every((key) => String(row[key] ?? "").trim() !== ""));

  const analysisProjectOptions = useMemo(() => results.map((result) => ({
    code: result.projectCode,
    label: result.projectName ? `${result.projectCode} — ${result.projectName}` : result.projectCode,
  })), [results]);

  const selectedAnalysisResult = useMemo(
    () => results.find((result) => result.projectCode === analysisProjectCode) || results[0] || null,
    [results, analysisProjectCode],
  );

  const projectDistribution = useMemo(() => {
    const empty = { high: 0, medium: 0, low: 0 };
    if (!selectedAnalysisResult) return { cost: empty, time: empty, overall: empty };
    const one = (key) => {
      const level = riskClass(selectedAnalysisResult[key]);
      return { high: level === "high" ? 1 : 0, medium: level === "medium" ? 1 : 0, low: level === "low" ? 1 : 0 };
    };
    return { cost: one("costRisk"), time: one("timeRisk"), overall: one("overallRisk") };
  }, [selectedAnalysisResult]);

  const historyResults = useMemo(
    () => results.filter((result) => (result.overrun?.quarterly?.length || 0) > 0 || (result.anomaly?.quarterly?.length || 0) > 0),
    [results],
  );

  const selectedHistoryResult = useMemo(
    () => historyResults.find((result) => result.projectCode === selectedProjectCode) || historyResults[0] || null,
    [historyResults, selectedProjectCode],
  );

  const timeSeries = useMemo(() => {
    if (!selectedHistoryResult) return { cost: [], time: [], anomaly: [] };

    const byDate = new Map();
    selectedHistoryResult.overrun?.quarterly?.forEach((point) => {
      const date = point.reporting_period_date;
      if (!date) return;
      const entry = byDate.get(date) || { date };
      if (point.cost_overrun?.predicted_overrun_pct !== null && point.cost_overrun?.predicted_overrun_pct !== undefined) {
        entry.cost = { date, value: Number(point.cost_overrun.predicted_overrun_pct) };
      }
      if (point.time_overrun?.predicted_overrun_pct !== null && point.time_overrun?.predicted_overrun_pct !== undefined) {
        entry.time = { date, value: Number(point.time_overrun.predicted_overrun_pct) };
      }
      byDate.set(date, entry);
    });
    selectedHistoryResult.anomaly?.quarterly?.forEach((point) => {
      const date = point.reporting_period_date;
      if (!date) return;
      const entry = byDate.get(date) || { date };
      if (point.anomaly_score !== null && point.anomaly_score !== undefined) {
        entry.anomaly = { date, value: Number(point.anomaly_score) };
      }
      byDate.set(date, entry);
    });

    const values = [...byDate.values()].sort((a, b) => String(a.date).localeCompare(String(b.date)));
    return {
      cost: values.filter((point) => point.cost).map((point) => point.cost),
      time: values.filter((point) => point.time).map((point) => point.time),
      anomaly: values.filter((point) => point.anomaly).map((point) => point.anomaly),
    };
  }, [selectedHistoryResult]);

  function updateRow(index, key, value) {
    const updater = (current) => current.map((row, rowIndex) => rowIndex === index ? { ...row, [key]: value } : row);
    if (mode === "single") {
      setSingleRecord((current) => ({ ...current, [key]: value }));
      setRows((current) => updater(current));
    } else if (mode === "batch") {
      setBatchRows(updater);
      setRows(updater);
    } else {
      setRows(updater);
    }
    setValidation([]);
  }

  function addRow() {
    setBatchRows((current) => [...current, emptyRecord()]);
    setRows((current) => [...current, emptyRecord()]);
  }

  function removeRow(index) {
    if (mode === "batch") {
      setBatchRows((current) => current.length === 1 ? current : current.filter((_, rowIndex) => rowIndex !== index));
    }
    setRows((current) => current.length === 1 ? current : current.filter((_, rowIndex) => rowIndex !== index));
  }

  function clearInputs() {
    if (mode === "single") {
      const blank = emptyRecord();
      setSingleRecord(blank);
      setRows([blank]);
    } else if (mode === "batch") {
      const blankRows = [emptyRecord()];
      setBatchRows(blankRows);
      setRows(blankRows);
    } else {
      setRows([]);
      setCsvName("");
      setCsvFile(null);
      if (fileRef.current) fileRef.current.value = "";
    }
    setResults([]);
    setSelectedResult(null);
    setSelectedProjectCode("");
    setAnalysisProjectCode("");
    setValidation([]);
    setApiState({ type: "idle", message: "" });
    setSummaryState({ type: "idle", text: "", model: "", warnings: [] });
  }

  function validateRows(dataRows = rows) {
    const errors = [];
    dataRows.forEach((row, index) => {
      REQUIRED_FIELD_KEYS.forEach((key) => {
        if (String(row[key] ?? "").trim() === "") {
          const label = FIELD_DEFINITIONS.find((field) => field.key === key)?.label || key;
          errors.push(`Row ${index + 1}: ${label} is required.`);
        }
      });
    });
    return errors;
  }

  function switchMode(nextMode) {
    setMode(nextMode);
    setValidation([]);
    setApiState({ type: "idle", message: "" });
    setSummaryState({ type: "idle", text: "", model: "", warnings: [] });
    setResults([]);
    setSelectedResult(null);
    setSelectedProjectCode("");
    setAnalysisProjectCode("");

    if (nextMode === "single") setRows([singleRecord]);
    if (nextMode === "batch") setRows(batchRows.length ? batchRows : [emptyRecord()]);
    if (nextMode === "csv") setRows([]);
  }

  function handleCsv(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setCsvName(file.name);
    setCsvFile(file);
    setValidation([]);
    setResults([]);
    setSummaryState({ type: "idle", text: "", model: "", warnings: [] });
    setApiState({ type: "idle", message: "" });

    const reader = new FileReader();
    reader.onload = () => {
      const parsed = parseCsv(String(reader.result || ""));
      if (!parsed.length) {
        setValidation(["The CSV is empty or could not be parsed."]);
        return;
      }

      const headers = parsed[0].map((header) => HEADER_ALIASES[header.trim().toLowerCase()] || null);
      const missing = BACKEND_REQUIRED_COLUMNS.filter((backendColumn) => {
        const requiredUiKey = Object.entries(API_FIELD_BY_UI_KEY).find(([, apiKey]) => apiKey === backendColumn)?.[0];
        return !headers.includes(requiredUiKey);
      });

      if (missing.length) {
        const labels = missing.map((backendColumn) => {
          const uiKey = Object.entries(API_FIELD_BY_UI_KEY).find(([, apiKey]) => apiKey === backendColumn)?.[0];
          return FIELD_DEFINITIONS.find((field) => field.key === uiKey)?.label || backendColumn;
        });
        setValidation([
          `Missing required columns: ${labels.join(", ")}`,
          "The backend requires the same raw fields used by classification, regression, and anomaly detection.",
        ]);
        return;
      }

      const imported = parsed.slice(1).map((values) => Object.fromEntries(
        UI_COLUMNS.map((key) => [key, values[headers.indexOf(key)] ?? ""]),
      ));

      if (!imported.length) {
        setValidation(["The CSV contains headers but no project records."]);
        return;
      }

      setRows(imported);
      setMode("csv");
      setApiState({ type: "success", message: `${imported.length} record${imported.length === 1 ? "" : "s"} loaded. Review the preview, then run analysis.` });
    };
    reader.onerror = () => setValidation(["The CSV could not be read. Please try the file again."]);
    reader.readAsText(file);
  }

  function downloadTemplate() {
    const header = [...BACKEND_REQUIRED_COLUMNS, ...BACKEND_OPTIONAL_COLUMNS];
    const blob = new Blob([`${header.join(",")}\n`], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "nirmaan-risk-analysis-template.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function readError(response) {
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") return body.detail;
      if (Array.isArray(body?.detail)) return body.detail.map((item) => item?.msg || "Validation error").join(" ");
      return body?.message || `API returned HTTP ${response.status}.`;
    } catch {
      return `API returned HTTP ${response.status}.`;
    }
  }

  async function postJson(path, body) {
    const separator = path.includes("?") ? "&" : "?";
    const endpoint = `${API_BASE_URL}${path}${separator}preferred_model=${encodeURIComponent(PREFERRED_MODEL)}`;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const detail = await readError(response);
      const requestId = response.headers.get("x-request-id");
      throw new Error(`HTTP ${response.status}: ${detail}${requestId ? ` (Request ID: ${requestId})` : ""}`);
    }
    return response.json();
  }

  async function runAnalysis() {
    const errors = mode === "csv" && rows.length === 0
      ? ["Upload a CSV before running analysis."]
      : validateRows();

    if (errors.length) {
      setValidation(errors.slice(0, 12));
      setApiState({ type: "error", message: "Resolve the required fields before running the risk service." });
      return;
    }

    setApiState({
      type: "loading",
      message: `Sending ${rows.length} record${rows.length === 1 ? "" : "s"} to the classification pipeline…`,
    });
    setSummaryState({ type: "idle", text: "", model: "", warnings: [] });

    try {
      let payload;

      if (mode === "single") {
        payload = await postJson("/classification/predict", [toRawRow(rows[0])]);
      } else if (mode === "batch") {
        const grouped = new Map();
        rows.forEach((row) => {
          const raw = toRawRow(row);
          const code = String(raw.project_code || "").trim();
          if (!grouped.has(code)) grouped.set(code, []);
          grouped.get(code).push(raw);
        });

        payload = await postJson("/classification/predict/batch", {
          projects: [...grouped.entries()].map(([project_code, projectRows]) => ({
            project_code,
            rows: projectRows,
          })),
        });
      } else {
        if (!csvFile) throw new Error("The selected CSV file is no longer available. Please choose it again.");
        const form = new FormData();
        form.append("file", csvFile);
        const response = await fetch(`${API_BASE_URL}/classification/predict/csv?preferred_model=${encodeURIComponent(PREFERRED_MODEL)}`, {
          method: "POST",
          body: form,
        });
        if (!response.ok) {
          const detail = await readError(response);
          const requestId = response.headers.get("x-request-id");
          throw new Error(`HTTP ${response.status}: ${detail}${requestId ? ` (Request ID: ${requestId})` : ""}`);
        }
        payload = await response.json();
      }

      const normalized = normalizePayload(payload, rows);
      if (!normalized.length) throw new Error("The backend returned no usable project predictions.");

      setResults(normalized);
      setSelectedProjectCode(normalized.find((item) => (item.overrun?.quarterly?.length || 0) > 0)?.projectCode || normalized[0].projectCode);
      setSelectedResult(null);
      setApiState({
        type: "success",
        message: `Analysis completed for ${normalized.length} project${normalized.length === 1 ? "" : "s"} using classification → gated regression → anomaly detection.`,
      });
    } catch (error) {
      setResults([]);
      setSelectedResult(null);
      setApiState({ type: "error", message: error.message || "The classification pipeline could not be reached." });
    }
  }

  async function generateSummary() {
    if (!results.length) return;

    setSummaryState({ type: "loading", text: "", model: "", warnings: [] });
    try {
      if (results.length === 1) {
        const result = await postJson("/summary/generate", {
          prediction: results[0].raw,
          project_name: results[0].projectName || undefined,
        });
        setSummaryState({
          type: "success",
          text: result.summary,
          model: result.model,
          warnings: result.warnings || [],
        });
        return;
      }

      const response = await postJson("/summary/generate/batch", {
        predictions: results.map((result) => result.raw),
      });

      const text = (response.summaries || [])
        .map((summary) => `${summary.project_code || "Project"}: ${summary.summary}`)
        .join("\n\n");

      setSummaryState({
        type: "success",
        text,
        model: response.model,
        warnings: (response.summaries || []).flatMap((summary) => summary.warnings || []),
      });
    } catch (error) {
      setSummaryState({ type: "error", text: error.message || "The GenAI summary service could not be reached.", model: "", warnings: [] });
    }
  }

  const formSections = [
    ["Project Information", "Project identity and administrative classification.", FIELD_DEFINITIONS.filter((field) => field.group === "project")],
    ["Financial Information", "Quarter and expenditure values used by the risk model.", FIELD_DEFINITIONS.filter((field) => field.group === "financial")],
    ["Timeline Information", "Approved and anticipated commissioning milestones.", FIELD_DEFINITIONS.filter((field) => field.group === "timeline")],
  ];

  const latestTableText = (result) => {
    const costPct = result.overrun?.cost_overrun?.predicted_overrun_pct;
    const timePct = result.overrun?.time_overrun?.predicted_overrun_pct;
    return {
      cost: costPct === null || costPct === undefined ? "NA" : `${numberValue(costPct, 1)}%`,
      time: timePct === null || timePct === undefined ? "NA" : `${numberValue(timePct, 1)}%`,
    };
  };

  const selectedQuarterRows = useMemo(() => {
    if (!selectedHistoryResult) return [];
    const byDate = new Map();
    (selectedHistoryResult.overrun?.quarterly || []).forEach((point) => {
      if (!point.reporting_period_date) return;
      byDate.set(point.reporting_period_date, {
        date: point.reporting_period_date,
        costProbability: null,
        timeProbability: null,
        costPct: point.cost_overrun?.predicted_overrun_pct ?? null,
        timePct: point.time_overrun?.predicted_overrun_pct ?? null,
        anomalyScore: null,
        anomalyRisk: null,
      });
    });
    (selectedHistoryResult.anomaly?.quarterly || []).forEach((point) => {
      if (!point.reporting_period_date) return;
      const current = byDate.get(point.reporting_period_date) || {
        date: point.reporting_period_date,
        costProbability: null,
        timeProbability: null,
        costPct: null,
        timePct: null,
        anomalyScore: null,
        anomalyRisk: null,
      };
      current.anomalyScore = point.anomaly_score ?? null;
      current.anomalyRisk = point.risk_level ?? null;
      byDate.set(point.reporting_period_date, current);
    });
    return [...byDate.values()].sort((a, b) => String(a.date).localeCompare(String(b.date)));
  }, [selectedHistoryResult]);

  return (
    <>
      <Navbar />
      <main className="risk-page">
        <header className="risk-page-header">
          <div>
            <div className="risk-heading-line"><span className="tricolor-line"><i /><i /><i /></span><span>IPMD • RISK ANALYSIS</span></div>
            <h1>Project Risk Analysis</h1>
            <p>Run the live PAIMANA classification pipeline, review gated cost/time overrun and anomaly results, inspect project history, and generate an AI summary.</p>
          </div>
        </header>

        <section className="risk-section mode-section">
          <div className="section-heading">
            <div><span className="section-kicker">STEP 01</span><h2>Select an input mode</h2><p>Select how project records will enter the risk workflow.</p></div>
            <div className="mode-tabs" role="tablist" aria-label="Risk input modes">
              {[["single", "Single Project"], ["batch", "Batch Input"], ["csv", "CSV Upload"]].map(([value, label]) => (
                <button key={value} className={mode === value ? "mode-tab active" : "mode-tab"} onClick={() => switchMode(value)} role="tab" aria-selected={mode === value}>{label}</button>
              ))}
            </div>
          </div>

          {mode === "csv" ? (
            <div className="csv-workflow">
              <div className="csv-drop" onClick={() => fileRef.current?.click()} role="button" tabIndex={0} onKeyDown={(event) => event.key === "Enter" && fileRef.current?.click()}>
                <div className="upload-symbol">↑</div>
                <div><strong>{csvName || "Upload a project CSV"}</strong><span>CSV only • backend raw-schema columns are validated before analysis</span></div>
                <button className="btn btn-secondary btn-sm" type="button">Choose CSV</button>
                <button className="btn btn-ghost btn-sm" type="button" onClick={(event) => { event.stopPropagation(); clearInputs(); }}>Clear</button>
                <input ref={fileRef} type="file" accept=".csv,text/csv" onChange={handleCsv} hidden />
              </div>
              <div className="csv-actions"><button className="text-button" onClick={downloadTemplate}>Download CSV template</button><span>Multiple rows for the same project_code are treated as its reporting history.</span></div>
            </div>
          ) : (
            <div className="mode-explainer">
              <div>
                <strong>{mode === "single" ? "One project record" : "Multiple project records or project history"}</strong>
                <span>{mode === "single" ? "Use the grouped form below for one project." : "Add as many rows as needed. Duplicate project codes are allowed and are sent together as one project's history."}</span>
              </div>
              {mode === "batch" && <button className="btn btn-secondary btn-sm" onClick={addRow}>+ Add project row</button>}
            </div>
          )}
        </section>

        {mode !== "csv" || rows.length ? (
          <section className="risk-section data-section">
            <div className="section-heading compact-heading">
              <div>
                <span className="section-kicker">STEP 02</span>
                <h2>{mode === "batch" || mode === "csv" ? "Project records" : "Project information"}</h2>
                <p>{mode === "csv" ? "Review imported values. The original CSV file is sent to the backend CSV endpoint." : "Complete the raw model fields. Duplicate project codes are supported in batch mode for historical scoring."}</p>
              </div>
              <div className="data-heading-actions">
                {mode === "batch" && <span className="record-count">{rows.length} record{rows.length === 1 ? "" : "s"}</span>}
                <button className="btn btn-ghost btn-sm" type="button" onClick={clearInputs}>Clear inputs</button>
              </div>
            </div>

            {mode === "batch" || mode === "csv" ? (
              <div className="table-scroll input-table-scroll">
                <table className="risk-table input-table">
                  <thead><tr><th>#</th>{FIELD_DEFINITIONS.map((field) => <th key={field.key}>{field.label}{field.required ? " *" : " (Optional)"}</th>)}<th>Action</th></tr></thead>
                  <tbody>{rows.map((row, index) => <tr key={row._rowId || `batch-row-${index}`}>
                    <td className="row-number">{index + 1}</td>
                    {FIELD_DEFINITIONS.map((field) => (
                      <td key={field.key}>
                        {field.type === "select" ? (
                          <select value={row[field.key] || ""} onChange={(event) => updateRow(index, field.key, event.target.value)}>
                            <option value="">Select</option>
                            {field.options.map((option) => <option key={option} value={option}>{option}</option>)}
                          </select>
                        ) : (
                          <input type={field.type} value={row[field.key] || ""} placeholder={field.placeholder || "—"} step={field.step} onChange={(event) => updateRow(index, field.key, event.target.value)} />
                        )}
                      </td>
                    ))}
                    <td><button className="remove-button" onClick={() => removeRow(index)} disabled={rows.length === 1} aria-label={`Remove row ${index + 1}`}>Remove</button></td>
                  </tr>)}</tbody>
                </table>
              </div>
            ) : (
              <div className="form-sections">
                {formSections.map(([title, description, fields]) => (
                  <div className="form-group" key={title}>
                    <div className="form-group-heading"><h3>{title}</h3><span>{description}</span></div>
                    <div className="field-grid">{fields.map((field) => <Field key={field.key} definition={field} value={rows[0][field.key]} onChange={(key, value) => updateRow(0, key, value)} />)}</div>
                  </div>
                ))}
              </div>
            )}
          </section>
        ) : null}

        <section className="risk-section prediction-section">
          <div className="section-heading compact-heading">
            <div><span className="section-kicker">STEP 03</span><h2>Run risk assessment</h2><p>Submit the same raw record/history to the classification pipeline; regression and anomaly legs are gated by the classifier.</p></div>
            <div className="prediction-actions">
              <span className={canAnalyze ? "validation-chip ready" : "validation-chip"}>{canAnalyze ? "All required fields complete" : "Required fields pending"}</span>
              <button className="btn btn-primary" onClick={runAnalysis} disabled={apiState.type === "loading"}>{apiState.type === "loading" ? "Running…" : "Run Risk Analysis"}</button>
            </div>
          </div>
          {apiState.message && <div className={`api-notice ${apiState.type}`}>{apiState.message}</div>}
          {validation.length > 0 && <div className="validation-box"><strong>Please check the following:</strong><ul>{validation.map((error) => <li key={error}>{error}</li>)}</ul></div>}
        </section>

        <section className="risk-section results-section">
          <div className="section-heading compact-heading">
            <div>
              <span className="section-kicker">STEP 04</span>
              <h2>Risk results</h2>
              <p>{results.length ? `${results.length} project${results.length === 1 ? "" : "s"} returned by the live classification pipeline, including gated regression/anomaly detail.` : "Prediction results will appear here after a successful model response."}</p>
            </div>
            <div className="section-heading-actions">
              {results.length > 0 && <span className="result-count">{results.length} project{results.length === 1 ? "" : "s"}</span>}
              {results.length > 0 && <button className="btn btn-secondary btn-sm" onClick={generateSummary} disabled={summaryState.type === "loading"}>{summaryState.type === "loading" ? "Generating…" : "Generate Summary"}</button>}
            </div>
          </div>

          {results.length === 0 ? (
            <div className="empty-results"><RiskIcon /><strong>No prediction results yet</strong><span>Run one of the live classification endpoints. No synthetic risk scores are shown.</span></div>
          ) : (
            <div className="table-scroll">
              <table className="risk-table results-table">
                <thead><tr><th>Project Code</th><th>Project Name</th><th>Cost Risk</th><th>Time Risk</th><th>Overall Risk</th><th>Cost Overrun</th><th>Time Overrun</th><th>Reason</th><th>Details</th></tr></thead>
                <tbody>
                  {results.map((result) => {
                    const latest = latestTableText(result);
                    return (
                      <tr key={result.id}>
                        <td><strong>{displayValue(result.projectCode)}</strong></td>
                        <td>{displayValue(result.projectName)}</td>
                        <td><RiskBadge value={`${result.costRisk} • ${percentValue(result.costProbability)}`} /></td>
                        <td><RiskBadge value={`${result.timeRisk} • ${percentValue(result.timeProbability)}`} /></td>
                        <td><RiskBadge value={result.overallRisk} /></td>
                        <td>{latest.cost}</td>
                        <td>{latest.time}</td>
                        <td className="reason-cell">{displayValue(result.reason.join(" "))}</td>
                        <td><button className="details-button" onClick={() => setSelectedResult(result)}>View</button></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {summaryState.type !== "idle" && (
            <div className={`summary-panel ${summaryState.type}`}>
              <div className="summary-panel-header">
                <div><span className="section-kicker">GENAI SUMMARY</span><h3>{summaryState.type === "loading" ? "Generating analysis summary…" : "AI interpretation of the returned model results"}</h3></div>
                {summaryState.model && <span className="summary-model">{summaryState.model}</span>}
              </div>
              {summaryState.text && summaryState.type !== "error" && <div className="summary-text">{summaryState.text.split("\n").map((paragraph, index) => <p key={`${paragraph}-${index}`}>{paragraph}</p>)}</div>}
              {summaryState.warnings.length > 0 && <div className="summary-warnings">{summaryState.warnings.map((warning) => <span key={warning}>{warning}</span>)}</div>}
              {summaryState.type === "error" && <p className="summary-error">{summaryState.text}</p>}
            </div>
          )}
        </section>

        <section className="risk-section analysis-section">
          <div className="section-heading compact-heading">
            <div><span className="section-kicker">STEP 05</span><h2>Risk analysis</h2><p>Use returned probabilities for portfolio distributions and returned quarterly regression/anomaly rows for time-series analysis.</p></div>
          </div>

          {results.length === 0 ? (
            <div className="analysis-placeholder">Charts and analytical tables become active after the ML pipeline returns prediction results.</div>
          ) : (
            <>
              <div className="analysis-project-toolbar">
                <div>
                  <span className="section-kicker">PROJECT VIEW</span>
                  <strong>{selectedAnalysisResult?.projectName || selectedAnalysisResult?.projectCode || "Project"}</strong>
                  <span>All analysis cards below are scoped to the selected project.</span>
                </div>
                <select value={selectedAnalysisResult?.projectCode || ""} onChange={(event) => { setAnalysisProjectCode(event.target.value); setSelectedProjectCode(event.target.value); }} aria-label="Select project for risk analysis">
                  {analysisProjectOptions.map((project) => <option key={project.code} value={project.code}>{project.label}</option>)}
                </select>
              </div>

              <div className="analysis-grid">
                {[
                  ["Cost-risk distribution", projectDistribution.cost, selectedAnalysisResult?.costRisk],
                  ["Time-risk distribution", projectDistribution.time, selectedAnalysisResult?.timeRisk],
                  ["Overall risk distribution", projectDistribution.overall, selectedAnalysisResult?.overallRisk],
                ].map(([title, distribution, currentRisk]) => {
                  const max = 1;
                  return (
                    <div className={`chart-panel distribution-card ${riskClass(currentRisk)}`} key={title}>
                      <div className="chart-title"><h3>{title}</h3><RiskBadge value={currentRisk} /></div>
                      <div className="distribution-subtitle">Current project classification</div>
                      {[
                        ["High", "high", "Elevated risk"],
                        ["Medium", "medium", "Moderate risk"],
                        ["Low", "low", "Lower risk"],
                      ].map(([label, level, hint]) => (
                        <div className="bar-row" key={level}>
                          <span>{label}</span>
                          <div className="bar-track"><i className={`bar-fill ${level}`} style={{ width: `${(distribution[level] / max) * 100}%` }} /></div>
                          <strong>{distribution[level]}</strong>
                          <small>{distribution[level] ? hint : "Not selected"}</small>
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>

              <div className="analysis-history">
                <div className="analysis-history-header">
                  <div>
                    <span className="section-kicker">HISTORY</span>
                    <h3>Quarterly risk trend</h3>
                    <p>Batch and CSV responses return row-wise regression/anomaly history whenever those services were gated on. Single-record inputs may only have one latest result.</p>
                  </div>
                  {historyResults.length > 0 && (
                    <select value={selectedHistoryResult?.projectCode || ""} onChange={(event) => setSelectedProjectCode(event.target.value)} aria-label="Select project history">
                      {historyResults.map((result) => <option key={result.projectCode} value={result.projectCode}>{result.projectCode}</option>)}
                    </select>
                  )}
                </div>

                {!selectedHistoryResult ? (
                  <div className="analysis-placeholder">No quarterly regression/anomaly series was returned. This can happen when the classifier probability is at or below the 0.5 gate, so gated regression and anomaly detection are intentionally skipped.</div>
                ) : (
                  <>
                    <div className="trend-grid">
                      <div className="chart-panel">
                        <div className="chart-title"><h3>Predicted cost / time overrun</h3><span>{selectedQuarterRows.length} quarters</span></div>
                        <LineChart
                          series={[
                            { key: "cost", label: "Cost overrun %", values: timeSeries.cost },
                            { key: "time", label: "Time overrun %", values: timeSeries.time },
                          ]}
                          valueFormatter={(value) => `${Number(value).toFixed(0)}%`}
                          emptyLabel="No gated regression series is available for this project."
                        />
                      </div>
                      <div className="chart-panel">
                        <div className="chart-title"><h3>Anomaly score by quarter</h3><span>{timeSeries.anomaly.length} points</span></div>
                        <LineChart
                          series={[{ key: "anomaly", label: "Anomaly score", values: timeSeries.anomaly }]}
                          valueFormatter={(value) => Number(value).toFixed(2)}
                          emptyLabel="No quarterly anomaly series is available for this project."
                        />
                      </div>
                    </div>

                    <div className="table-scroll history-table-scroll">
                      <table className="risk-table history-table">
                        <thead><tr><th>Reporting Period</th><th>Cost Overrun</th><th>Time Overrun</th><th>Anomaly Score</th><th>Anomaly Risk</th></tr></thead>
                        <tbody>
                          {selectedQuarterRows.map((row) => (
                            <tr key={row.date}>
                              <td><strong>{displayValue(row.date)}</strong></td>
                              <td>{row.costPct === null ? "NA" : `${numberValue(row.costPct, 2)}%`}</td>
                              <td>{row.timePct === null ? "NA" : `${numberValue(row.timePct, 2)}%`}</td>
                              <td>{numberValue(row.anomalyScore, 3)}</td>
                              <td>{row.anomalyRisk ? <RiskBadge value={row.anomalyRisk} /> : "NA"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </section>
      </main>

      {selectedResult && (
        <div className="drawer-backdrop" onMouseDown={() => setSelectedResult(null)}>
          <aside className="result-drawer" onMouseDown={(event) => event.stopPropagation()}>
            <div className="drawer-header">
              <div><span className="section-kicker">PROJECT RESULT</span><h2>{displayValue(selectedResult.projectName)}</h2><p>{displayValue(selectedResult.projectCode)} • {displayValue(selectedResult.agencyName)}</p></div>
              <button className="drawer-close" onClick={() => setSelectedResult(null)} aria-label="Close details">×</button>
            </div>

            <div className="drawer-risks">
              <div><span>Cost Classification</span><RiskBadge value={`${selectedResult.costRisk} • ${percentValue(selectedResult.costProbability)}`} /></div>
              <div><span>Time Classification</span><RiskBadge value={`${selectedResult.timeRisk} • ${percentValue(selectedResult.timeProbability)}`} /></div>
              <div><span>Overall Risk</span><RiskBadge value={selectedResult.overallRisk} /></div>
            </div>

            <div className="drawer-block">
              <h3>Why the model flagged it</h3>
              <p><strong>Cost:</strong> {displayValue(selectedResult.costReason)}</p>
              <p><strong>Time:</strong> {displayValue(selectedResult.timeReason)}</p>
            </div>


            <div className="drawer-block">
              <h3>Overrun forecast</h3>
              <dl>
                <dt>Cost overrun</dt><dd>{selectedResult.overrun?.cost_overrun?.predicted_overrun_pct == null ? "NA" : `${numberValue(selectedResult.overrun?.cost_overrun?.predicted_overrun_pct, 2)}%`}</dd>
                <dt>Final cost</dt><dd>{numberValue(selectedResult.overrun?.cost_overrun?.predicted_final_cost_rs_cr, 2)} ₹ Cr</dd>
                <dt>Cost increase</dt><dd>{numberValue(selectedResult.overrun?.cost_overrun?.predicted_cost_increase_rs_cr, 2)} ₹ Cr</dd>
                <dt>Time overrun</dt><dd>{selectedResult.overrun?.time_overrun?.predicted_overrun_pct == null ? "NA" : `${numberValue(selectedResult.overrun?.time_overrun?.predicted_overrun_pct, 2)}%`}</dd>
                <dt>Delay</dt><dd>{numberValue(selectedResult.overrun?.time_overrun?.predicted_delay_months, 2)} months</dd>
                <dt>Forecast date</dt><dd>{displayValue(selectedResult.overrun?.time_overrun?.predicted_final_commissioning_date)}</dd>
              </dl>
              {(selectedResult.overrun?.cost_overrun?.reason || selectedResult.overrun?.time_overrun?.reason) && (
                <p className="drawer-note">{displayValue(selectedResult.overrun?.cost_overrun?.reason || selectedResult.overrun?.time_overrun?.reason)}</p>
              )}
            </div>

            <div className="drawer-block">
              <h3>Anomaly assessment</h3>
              <dl>
                <dt>Risk level</dt><dd>{displayValue(selectedResult.anomaly?.risk_level)}</dd>
                <dt>Anomaly score</dt><dd>{numberValue(selectedResult.anomaly?.anomaly_score, 3)}</dd>
              </dl>
              <p className="drawer-note">{displayValue(selectedResult.anomaly?.anomaly_reason)}</p>
            </div>

            <div className="drawer-block">
              <h3>Project context</h3>
              <dl>
                <dt>Sector</dt><dd>{displayValue(selectedResult.sector)}</dd>
                <dt>State / UT</dt><dd>{displayValue(selectedResult.state)}</dd>
                <dt>Agency</dt><dd>{displayValue(selectedResult.agencyName)}</dd>
                <dt>Rows used</dt><dd>{displayValue(selectedResult.rowsUsed)}</dd>
              </dl>
            </div>
          </aside>
        </div>
      )}
    </>
  );
}

export default RiskAnalysis;
