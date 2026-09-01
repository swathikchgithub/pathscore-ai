"use client";

import { useEffect, useState } from "react";
import { getLeaderboard, getUseCases } from "@/lib/api";
import type { LeaderboardEntry, UseCaseInfo } from "@/lib/types";
import Leaderboard from "@/components/Leaderboard";
import ShapBarChart from "@/components/ShapBarChart";

const ENTITY_LABEL: Record<string, string> = {
  contact: "contacts",
  account: "accounts",
};

export default function Home() {
  const [useCases, setUseCases] = useState<UseCaseInfo[]>([]);
  const [selectedUseCase, setSelectedUseCase] = useState<string>("");
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [selectedEntry, setSelectedEntry] = useState<LeaderboardEntry | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getUseCases()
      .then((cases) => {
        setUseCases(cases);
        if (cases.length > 0) setSelectedUseCase(cases[0].name);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selectedUseCase) return;
    // Guards against an out-of-order response: if the use case changes again
    // before this fetch resolves, the cleanup below flips `cancelled` before
    // the stale .then() can overwrite newer data with older data.
    let cancelled = false;
    setLoading(true);
    setError(null);
    setSelectedEntry(null);
    getLeaderboard(selectedUseCase, 25)
      .then((data) => {
        if (cancelled) return;
        setEntries(data.results);
        setSelectedEntry(data.results[0] ?? null);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedUseCase]);

  const activeUseCase = useCases.find((uc) => uc.name === selectedUseCase);
  const entityLabel = activeUseCase ? ENTITY_LABEL[activeUseCase.entity] ?? activeUseCase.entity : "";
  // Falls back to the raw index for use cases with no named classes (every
  // binary use case today) -- only Funnel Stage currently declares stage_names.
  const classLabel = (cls: string) => activeUseCase?.class_labels?.[cls] ?? `class ${cls}`;
  // score_pct is the one field every use case shares (binary or multi-class),
  // so it's what drives the detail panel's tier color -- high/low use the
  // same --positive/--negative tokens as the SHAP bars, mid stays accent.
  const scoreTier = (pct: number) => (pct >= 66 ? "high" : pct < 33 ? "low" : "mid");

  return (
    <main className="page">
      <header className="header">
        <h1>PathScore AI</h1>
        <p className="subtitle">
          A shared GTM scoring pipeline. Pick a use case to see who ranks highest and why.
        </p>
        <select
          value={selectedUseCase}
          onChange={(e) => setSelectedUseCase(e.target.value)}
          className="use-case-select"
        >
          {useCases.map((uc) => (
            <option key={uc.name} value={uc.name}>
              {uc.display_name}
            </option>
          ))}
        </select>

        {activeUseCase && (
          <div className="use-case-info">
            <p className="use-case-description">{activeUseCase.description}</p>
            <p className="use-case-meta">
              Scores <strong>{entityLabel}</strong> &middot; predicts{" "}
              <code>{activeUseCase.label_column}</code>
            </p>
          </div>
        )}
      </header>

      <details className="how-it-works" open>
        <summary>How does scoring work?</summary>
        <ol>
          <li>
            <strong>Score</strong> — a LightGBM model trained on this use case&apos;s historical
            outcomes (e.g. which contacts actually converted) outputs a probability for each
            entity, calibrated so a &ldquo;70%&rdquo; score reflects real-world odds, not a raw
            model guess.
          </li>
          <li>
            <strong>Rank</strong> — every entity in the sample is scored the same way, then sorted
            highest to lowest. There&apos;s no separate ranking step or model — rank 1 is simply
            whoever scored highest.
          </li>
          <li>
            <strong>Predicted class</strong> — the model estimates a probability for every possible
            outcome (see &ldquo;Class probabilities&rdquo; when you select a row); &ldquo;Predicted&rdquo;
            is whichever one is most likely, not just &ldquo;above 50%&rdquo; — that distinction
            matters most for multi-class use cases like Funnel Stage.
          </li>
          <li>
            <strong>Top factors</strong> — every prediction ships with the specific features that
            drove that individual score (SHAP values), not a generic importance ranking — what
            actually mattered for this one entity.
          </li>
        </ol>
        <p>
          Labels come from real historical outcomes, not editorial guesses, and an automated check
          blocks any feature that&apos;s secretly a stand-in for the outcome itself before a model
          is ever trained (see leakage_checks.py). This demo runs on synthetic GTM data standing in
          for a real warehouse — see docs/ below for the full technical design.
        </p>
      </details>

      {error && <div className="error">{error}</div>}
      {loading && <div className="loading">Loading...</div>}

      <div className="content">
        <div>
          <p className="section-caption">
            Ranked highest to lowest by calibrated score, from a sample of {entityLabel || "entities"}{" "}
            due for scoring. Click a row to see why it scored that way.
          </p>
          <Leaderboard
            entries={entries}
            selectedId={selectedEntry?.id}
            onSelect={setSelectedEntry}
            classLabel={classLabel}
          />
        </div>
        <div className="detail-panel">
          {selectedEntry ? (
            // Re-keyed to the entry's id: a fresh DOM node per selection so
            // the flash animation (globals.css) actually restarts instead
            // of React quietly patching text in an existing node.
            <div
              key={selectedEntry.id}
              className={`detail-panel-content tier-${scoreTier(selectedEntry.score_pct)}`}
            >
              <h2>{selectedEntry.id}</h2>
              <p className="score">
                {selectedEntry.score_pct.toFixed(1)}% probability &middot; predicted:{" "}
                {classLabel(selectedEntry.predicted_class)}
              </p>

              <h3>Class probabilities</h3>
              <ul className="class-probs">
                {Object.entries(selectedEntry.class_probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([cls, prob]) => (
                    <li key={cls}>
                      <span className="class-probs-label">{classLabel(cls)}</span>
                      <span className="class-probs-value">{(prob * 100).toFixed(1)}%</span>
                    </li>
                  ))}
              </ul>

              <h3>Top factors</h3>
              <p className="section-caption">
                The features that most influenced this specific prediction (SHAP values) — green
                pushes the score up, red pulls it down.
              </p>
              <ShapBarChart factors={selectedEntry.top_factors} />
            </div>
          ) : (
            <p className="empty">Select a row to see score factors.</p>
          )}
        </div>
      </div>

      <footer className="footer">
        Scored against synthetic GTM data for this demo — see{" "}
        <a href="https://github.com/swathikchgithub/pathscore-ai/tree/main/docs">
          docs/
        </a>{" "}
        for the architecture, decision records, and full pipeline behind these numbers.
      </footer>
    </main>
  );
}
