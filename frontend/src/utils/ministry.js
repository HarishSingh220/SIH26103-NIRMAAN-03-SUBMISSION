// Sector -> Ministry lookup and aggregation helpers.
//
// The source dataset (projects-summary.json) records each project's
// *sector* (e.g. "RAILWAYS", "POWER") but not an explicit ministry name.
// This map translates the sector taxonomy already in the data into the
// administrative ministry that owns it, so Sector-wise and Ministry-wise
// views can both be derived from the same real project records instead
// of hardcoding project names or counts.
export const SECTOR_TO_MINISTRY = {
  RAILWAYS: "Ministry of Railways",
  "ROAD TRANSPORT AND HIGHWAYS": "Ministry of Road Transport & Highways",
  POWER: "Ministry of Power",
  PETROLEUM: "Ministry of Petroleum & Natural Gas",
  PETROCHEMICALS: "Dept. of Chemicals & Petrochemicals",
  "ATOMIC ENERGY": "Department of Atomic Energy",
  COAL: "Ministry of Coal",
  STEEL: "Ministry of Steel",
  MINES: "Ministry of Mines",
  "URBAN DEVELOPMENT": "Ministry of Housing & Urban Affairs",
  "CIVIL AVIATION": "Ministry of Civil Aviation",
  "SHIPPING AND PORTS": "Ministry of Ports, Shipping & Waterways",
  TELECOMMUNICATIONS: "Department of Telecommunications",
  "WATER RESOURCES": "Ministry of Jal Shakti",
  "HEALTH AND FAMILY WELFARE": "Ministry of Health & Family Welfare",
  "DEPARTMENT OF HIGHER EDUCATION": "Ministry of Education",
  "HOME AFFAIRS": "Ministry of Home Affairs",
  DPIIT: "Dept. for Promotion of Industry & Internal Trade",
};

export function ministryFor(sector) {
  return SECTOR_TO_MINISTRY[sector] || toTitleCase(sector);
}

/**
 * The dataset ships two overlapping project lists — `highValue` (the
 * top-ranked projects by cost) and `byState` (per-state top-project
 * records). Neither is the full 3,769-project set on its own, so this
 * merges them into one de-duplicated list keyed by project code for
 * views (like the high-value grid) that just need "some real projects",
 * not an exhaustive count.
 *
 * For anything that reports a total (sector/ministry breakdowns, state
 * counts), use `data.sectorBreakdown` / `data.states` directly instead —
 * those are precomputed from every project in the CSV by
 * scripts/build_data.py, not just this sampled subset.
 */
export function getAllProjects(data) {
  const map = new Map();
  Object.values(data.byState || {}).forEach((projects) => {
    projects.forEach((p) => map.set(p.code, p));
  });
  (data.highValue || []).forEach((p) => {
    if (!map.has(p.code)) map.set(p.code, p);
  });
  return Array.from(map.values());
}

/** Sector -> project count, sorted descending. Precomputed from all 3,769 projects. */
export function getSectorBreakdown(data) {
  return (data.sectorBreakdown || []).map((s) => ({ ...s }));
}

/** Ministry -> project count, sorted descending, aggregated from the (complete) sector breakdown. */
export function getMinistryBreakdown(data) {
  const sectorCounts = getSectorBreakdown(data);
  const counts = new Map();
  sectorCounts.forEach(({ sector, count }) => {
    const ministry = ministryFor(sector);
    counts.set(ministry, (counts.get(ministry) || 0) + count);
  });
  return Array.from(counts.entries())
    .map(([ministry, count]) => ({ ministry, count }))
    .sort((a, b) => b.count - a.count);
}

/** Top N projects by current (anticipated) cost, descending — used for the High Value cards. */
export function getTopHighValueProjects(data, n = 10) {
  return (data.highValue || []).slice(0, n);
}

function toTitleCase(s) {
  return s.replace(/\w\S*/g, (w) => w.charAt(0) + w.slice(1).toLowerCase());
}
