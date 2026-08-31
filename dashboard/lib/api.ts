import type { LeaderboardResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export async function getUseCases(): Promise<string[]> {
  const res = await fetch(`${API_BASE_URL}/use-cases`);
  if (!res.ok) throw new Error(`Failed to load use cases (${res.status})`);
  return res.json();
}

export async function getLeaderboard(useCase: string, limit = 25): Promise<LeaderboardResponse> {
  const res = await fetch(`${API_BASE_URL}/score/${useCase}/leaderboard?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to load leaderboard for ${useCase} (${res.status})`);
  return res.json();
}
