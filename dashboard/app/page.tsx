"use client";

import { useEffect, useState } from "react";
import { getLeaderboard, getUseCases } from "@/lib/api";
import type { LeaderboardEntry } from "@/lib/types";
import Leaderboard from "@/components/Leaderboard";
import ShapBarChart from "@/components/ShapBarChart";

export default function Home() {
  const [useCases, setUseCases] = useState<string[]>([]);
  const [selectedUseCase, setSelectedUseCase] = useState<string>("");
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [selectedEntry, setSelectedEntry] = useState<LeaderboardEntry | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getUseCases()
      .then((cases) => {
        setUseCases(cases);
        if (cases.length > 0) setSelectedUseCase(cases[0]);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selectedUseCase) return;
    setLoading(true);
    setError(null);
    setSelectedEntry(null);
    getLeaderboard(selectedUseCase, 25)
      .then((data) => {
        setEntries(data.results);
        setSelectedEntry(data.results[0] ?? null);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [selectedUseCase]);

  return (
    <main className="page">
      <header className="header">
        <h1>PathScore AI</h1>
        <p className="subtitle">Ranked scoring dashboard</p>
        <select
          value={selectedUseCase}
          onChange={(e) => setSelectedUseCase(e.target.value)}
          className="use-case-select"
        >
          {useCases.map((uc) => (
            <option key={uc} value={uc}>
              {uc}
            </option>
          ))}
        </select>
      </header>

      {error && <div className="error">{error}</div>}
      {loading && <div className="loading">Loading...</div>}

      <div className="content">
        <Leaderboard entries={entries} selectedId={selectedEntry?.id} onSelect={setSelectedEntry} />
        <div className="detail-panel">
          {selectedEntry ? (
            <>
              <h2>{selectedEntry.id}</h2>
              <p className="score">
                {selectedEntry.score_pct.toFixed(1)}% &middot; predicted: {selectedEntry.predicted_class}
              </p>
              <h3>Top factors</h3>
              <ShapBarChart factors={selectedEntry.top_factors} />
            </>
          ) : (
            <p className="empty">Select a row to see score factors.</p>
          )}
        </div>
      </div>
    </main>
  );
}
