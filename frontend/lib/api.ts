export type Signal = {
  id: number;
  lat: number;
  lng: number;
  city: string;
  state: string;
  region?: string;
  country?: string;
  category: string;
  title: string;
  description: string;
  budget: number | null;
  timeline: string;
  stakeholders: string[];
  source_url?: string;
};

export type MatchResult = {
  signal: Signal;
  score: number;
  reasoning: string;
};

export type MatchResponse = {
  matches: MatchResult[];
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export async function matchStartup(startupDescription: string) {
  const response = await fetch(`${API_BASE}/match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ startup_description: startupDescription }),
  });

  if (!response.ok) {
    throw new Error(`Match request failed: ${response.status}`);
  }

  return (await response.json()) as MatchResponse;
}
