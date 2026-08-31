import type { Factor } from "@/lib/types";

interface Props {
  factors: Factor[];
}

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
              {f.feature}
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
