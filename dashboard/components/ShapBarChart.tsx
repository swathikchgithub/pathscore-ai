import type { Factor } from "@/lib/types";

interface Props {
  factors: Factor[];
}

// Most feature names are self-explanatory as-is; this only exists for the
// one feature whose source isn't obvious from its name -- avg_intent_score
// is a Snowflake Cortex sentiment/intent extraction over event text, not a
// plain warehouse column like the rest.
const FEATURE_LABELS: Record<string, string> = {
  avg_intent_score: "Intent/sentiment score (Snowflake Cortex)",
};

export default function ShapBarChart({ factors }: Props) {
  const maxAbs = Math.max(...factors.map((f) => Math.abs(f.impact)), 0.0001);

  return (
    <div className="shap-chart">
      {factors.map((f) => {
        const widthPct = (Math.abs(f.impact) / maxAbs) * 100;
        const isPositive = f.impact >= 0;
        return (
          <div key={f.feature} className="shap-row">
            <span className="shap-label" title={f.feature}>
              {FEATURE_LABELS[f.feature] ?? f.feature}
            </span>
            <div className="shap-bar-track">
              <div
                className={`shap-bar ${isPositive ? "positive" : "negative"}`}
                style={{ width: `${widthPct}%` }}
              />
            </div>
            <span className="shap-value">{f.impact.toFixed(3)}</span>
          </div>
        );
      })}
    </div>
  );
}
