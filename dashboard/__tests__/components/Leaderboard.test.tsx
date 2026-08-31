import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import Leaderboard from "@/components/Leaderboard";
import type { LeaderboardEntry } from "@/lib/types";

const entries: LeaderboardEntry[] = [
  { id: "ACC-1", score_pct: 92.5, predicted_class: "1", class_probabilities: {}, top_factors: [] },
  { id: "ACC-2", score_pct: 41.2, predicted_class: "0", class_probabilities: {}, top_factors: [] },
];

describe("Leaderboard", () => {
  it("renders one row per entry with rank, id, score, and predicted class", () => {
    render(<Leaderboard entries={entries} onSelect={() => {}} />);

    const rows = screen.getAllByRole("row");
    expect(rows).toHaveLength(entries.length + 1); // + header row

    const [firstDataRow, secondDataRow] = rows.slice(1);
    expect(within(firstDataRow).getAllByRole("cell").map((c) => c.textContent)).toEqual([
      "1", // rank
      "ACC-1",
      "92.5%",
      "1", // predicted_class
    ]);
    expect(within(secondDataRow).getAllByRole("cell").map((c) => c.textContent)).toEqual([
      "2", // rank
      "ACC-2",
      "41.2%",
      "0", // predicted_class
    ]);
  });

  it("renders only the header row when there are no entries", () => {
    render(<Leaderboard entries={[]} onSelect={() => {}} />);
    expect(screen.getAllByRole("row")).toHaveLength(1);
  });

  it("marks the selected entry's row and leaves the others unmarked", () => {
    render(<Leaderboard entries={entries} selectedId="ACC-2" onSelect={() => {}} />);

    expect(screen.getByText("ACC-2").closest("tr")).toHaveClass("selected");
    expect(screen.getByText("ACC-1").closest("tr")).not.toHaveClass("selected");
  });

  it("calls onSelect with the clicked entry", async () => {
    const onSelect = vi.fn();
    render(<Leaderboard entries={entries} onSelect={onSelect} />);

    await userEvent.click(screen.getByText("ACC-2"));

    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith(entries[1]);
  });

  it("shows the raw predicted_class value when no classLabel prop is given", () => {
    render(<Leaderboard entries={entries} onSelect={() => {}} />);
    expect(screen.getByText("ACC-1").closest("tr")).toHaveTextContent("1");
  });

  it("maps predicted_class through classLabel when given (e.g. Funnel Stage's named classes)", () => {
    const stageNames: Record<string, string> = { "0": "MQL", "1": "SQL" };
    render(
      <Leaderboard entries={entries} onSelect={() => {}} classLabel={(c) => stageNames[c] ?? c} />
    );

    const firstDataRow = screen.getAllByRole("row")[1];
    const predictedCell = within(firstDataRow).getAllByRole("cell")[3];
    expect(predictedCell).toHaveTextContent("SQL"); // entries[0].predicted_class === "1"
  });
});
