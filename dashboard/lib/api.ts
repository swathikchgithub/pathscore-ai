import type { LeaderboardResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

// Only needed if the API is running with API_KEY set (src/serving/app.py) --
// unset on both sides by default, matching the open local-dev quickstart.
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;
const AUTH_HEADERS: HeadersInit = API_KEY ? { "X-API-Key": API_KEY } : {};

export async function getUseCases(): Promise<string[]> {
  const res = await fetch(`${API_BASE_URL}/use-cases`);
  if (!res.ok) throw new Error(`Failed to load use cases (${res.status})`);
  return res.json();
}

export async function getLeaderboard(useCase: string, limit = 25): Promise<LeaderboardResponse> {
  const res = await fetch(`${API_BASE_URL}/score/${useCase}/leaderboard?limit=${limit}`, {
    headers: AUTH_HEADERS,
  });
  if (!res.ok) throw new Error(`Failed to load leaderboard for ${useCase} (${res.status})`);
  return res.json();
}
