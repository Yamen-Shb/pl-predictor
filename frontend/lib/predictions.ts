// frontend/src/api/predictions.ts
const API_BASE = "http://localhost:8000";

export async function getLatestPredictions() {
  const res = await fetch(`${API_BASE}/predictions/latest`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}