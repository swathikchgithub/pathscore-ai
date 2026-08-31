export interface Factor {
  feature: string;
  impact: number;
}

export interface UseCaseInfo {
  name: string;
  display_name: string;
  description: string;
  entity: string;
  label_column: string;
  class_labels: Record<string, string> | null;
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
