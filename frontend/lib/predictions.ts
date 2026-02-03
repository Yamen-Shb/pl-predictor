// frontend/lib/predictions.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL;

export async function getLatestPredictions() {
  const res = await fetch(`${API_BASE}/predictions/latest`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}