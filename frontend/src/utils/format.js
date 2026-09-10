// Formatting helpers tuned for Indian crore-denominated infrastructure figures.

const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });

/** 108000 -> "₹1,08,000 Cr" | 892.4 -> "₹892 Cr" */
export function formatCr(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `₹${inr.format(Math.round(value))} Cr`;
}

/** 4479118 -> "₹44.79 lakh Cr" for headline totals */
export function formatLakhCr(value) {
  if (!value) return "—";
  return `₹${(value / 100000).toFixed(2)} lakh Cr`;
}

/** 445 -> "445" with Indian grouping */
export function formatCount(value) {
  return inr.format(value || 0);
}

export const STATUS_LABELS = {
  "on-track": "On Track",
  "minor-overrun": "Minor Overrun",
  "at-risk": "At Risk",
  reporting: "Reporting",
};
