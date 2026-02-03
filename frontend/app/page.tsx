"use client"

import { useEffect, useState } from "react"
import { getLatestPredictions } from "@/lib/predictions"
import { DashboardHeader } from "@/components/dashboard-header"
import { DashboardFooter } from "@/components/dashboard-footer"
import { MatchCard } from "@/components/match-card"

type Prediction = {
  match_id: string
  date: string
  home_team: string
  away_team: string
  home_pred: number
  away_pred: number
  home_win_pct: number
  draw_pct: number
  away_win_pct: number
}

type PredictionsResponse = {
  gameweek: number | null
  count: number
  predictions: Prediction[]
  generated_at: string
}

type Gameweek = {
  gameweek: number
}

export default function PredictionsDashboard() {
  const [currentGameweek, setCurrentGameweek] = useState<number | null>(null)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [lastUpdated, setLastUpdated] = useState("")

  useEffect(() => {
    async function fetchPredictions() {
      try {
        const data: PredictionsResponse = await getLatestPredictions()
        setPredictions(data.predictions)
        setCurrentGameweek(data.gameweek)
        setLastUpdated(new Date(data.generated_at).toLocaleString())
      } catch (err) {
        console.error("Failed to fetch predictions:", err)
      }
    }

    fetchPredictions()
  }, [])

  return (
    <div className="min-h-screen flex flex-col relative">
      <div className="fixed inset-0 bg-gradient-to-br from-[#0a0a0a] to-[#1a1a1a]" />
      <div className="fixed inset-0 pitch-pattern opacity-100 pointer-events-none" />
      <div className="fixed inset-0 bg-gradient-to-br from-[#3d195b]/15 via-transparent to-[#00ff85]/5 pointer-events-none" />
      <div className="fixed top-0 right-0 w-1/2 h-1/2 bg-gradient-to-bl from-[#04f5ff]/5 to-transparent pointer-events-none" />
      
      <div className="relative z-10 min-h-screen flex flex-col">
        <DashboardHeader gameweek={currentGameweek ?? 0} />

        <main className="flex-1 container mx-auto px-4 py-6 md:py-8">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 md:gap-6">
            {predictions.map((match, index) => (
              <MatchCard
                key={`${match.home_team}-${match.away_team}-${index}`}
                homeTeam={match.home_team}
                awayTeam={match.away_team}
                homeScore={match.home_pred}
                awayScore={match.away_pred}
                homeWinProb={Math.round(match.home_win_pct*100)}
                drawProb={Math.round(match.draw_pct*100)}
                awayWinProb={Math.round(match.away_win_pct*100)}
                matchDate={new Date(match.date).toLocaleDateString()}
                matchTime={new Date(match.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              />
            ))}
          </div>
        </main>

        <DashboardFooter lastUpdated={lastUpdated || "Loading..."} />
      </div>
    </div>
  )
}
