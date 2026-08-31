import type { LeaderboardEntry } from "@/lib/types";

interface Props {
  entries: LeaderboardEntry[];
  selectedId?: string;
  onSelect: (entry: LeaderboardEntry) => void;
  classLabel?: (cls: string) => string;
}

export default function Leaderboard({ entries, selectedId, onSelect, classLabel = (c) => c }: Props) {
  return (
    <table className="leaderboard">
      <thead>
        <tr>
          <th>Rank</th>
          <th>ID</th>
          <th>Score</th>
          <th>Predicted</th>
        </tr>
      </thead>
      <tbody>
        {entries.map((entry, i) => (
          <tr
            key={entry.id}
            className={entry.id === selectedId ? "selected" : ""}
            onClick={() => onSelect(entry)}
          >
            <td>{i + 1}</td>
            <td>{entry.id}</td>
            <td>{entry.score_pct.toFixed(1)}%</td>
            <td>{classLabel(entry.predicted_class)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
