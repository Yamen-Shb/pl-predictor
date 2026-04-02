import { supabase } from "@/lib/supabase"

export async function getLatestPredictions() {
  const { data, error } = await supabase
    .from("predictions_current")
    .select("*")
    .order("match_date", { ascending: true })

  if (error) throw new Error(error.message)

  const gameweek = data?.[0]?.gameweek ?? null

  return {
    gameweek,
    count: data?.length ?? 0,
    predictions: data ?? [],
    generated_at: new Date().toISOString(),
  }
}