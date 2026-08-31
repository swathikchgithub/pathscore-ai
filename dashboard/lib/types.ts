export interface Factor {
  feature: string;
  impact: number;
}

export interface LeaderboardEntry {
  id: string;
  score_pct: number;
  predicted_class: string;
  class_probabilities: Record<string, number>;
  top_factors: Factor[];
}

export interface LeaderboardResponse {
  use_case: string;
  count: number;
  results: LeaderboardEntry[];
}
