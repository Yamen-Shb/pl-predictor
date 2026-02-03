"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Calendar, Clock } from "lucide-react"
import { useEffect, useState } from "react"

// Team colors mapping
const teamColors: Record<string, string> = {
  "Arsenal FC": "#EF0107",
  "Chelsea FC": "#034694",
  "Liverpool FC": "#C8102E",
  "Manchester City FC": "#6CABDD",
  "Manchester United FC": "#DA291C",
  "Tottenham Hotspur FC": "#132257",
  "Newcastle United FC": "#241F20",
  "Brighton & Hove Albion FC": "#0057B8",
  "Aston Villa FC": "#670E36",
  "West Ham United FC": "#7A263A",
  "Everton FC": "#003399",
  "Wolverhampton Wanderers FC": "#FDB913",
  "AFC Bournemouth": "#DA291C",
  "Crystal Palace FC": "#1B458F",
  "Brentford FC": "#E30613",
  "Nottingham Forest FC": "#DD0000",
  "Fulham FC": "#000000",
  "Leeds United FC": "#FFCD00",
  "Burnley FC": "#6C1D45",
  "Sunderland AFC": "#FF0000",
  "Leicester": "#003090",
  "Ipswich": "#0044AA",
  "Southampton": "#D71920"
}

function getTeamColor(teamName: string): string {
  return teamColors[teamName] || "#666666"
}

interface MatchCardProps {
  homeTeam: string
  awayTeam: string
  homeScore: number
  awayScore: number
  homeWinProb: number
  drawProb: number
  awayWinProb: number
  matchDate: string
  matchTime: string
}

export function MatchCard({
  homeTeam,
  awayTeam,
  homeScore,
  awayScore,
  homeWinProb,
  drawProb,
  awayWinProb,
  matchDate,
  matchTime,
}: MatchCardProps) {
  const [isVisible, setIsVisible] = useState(false)
  
  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 100)
    return () => clearTimeout(timer)
  }, [])

  const homeColor = getTeamColor(homeTeam)
  const awayColor = getTeamColor(awayTeam)

  return (
    <Card className="group relative bg-card border-border hover:border-[#3d195b]/60 transition-all duration-300 overflow-hidden">
      {/* Premier League gradient hover effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#3d195b]/0 via-[#00ff85]/0 to-[#04f5ff]/0 group-hover:from-[#3d195b]/10 group-hover:via-[#00ff85]/5 group-hover:to-[#04f5ff]/10 transition-all duration-500 pointer-events-none" />
      
      <CardContent className="relative p-0">
        {/* Match Date/Time Header */}
        <div className="flex items-center justify-between px-4 py-2 bg-secondary/50 border-b border-border">
          <div className="flex items-center gap-2 text-muted-foreground text-sm">
            <Calendar className="w-3.5 h-3.5" />
            <span>{matchDate}</span>
          </div>
          <div className="flex items-center gap-2 text-muted-foreground text-sm">
            <Clock className="w-3.5 h-3.5" />
            <span>{matchTime}</span>
          </div>
        </div>

        {/* Teams and Score */}
        <div className="p-4">
          <div className="flex items-center justify-between gap-3">
            {/* Home Team */}
            <div className="flex-1 flex flex-col items-center gap-2">
              <div 
                className="w-12 h-12 rounded-full bg-secondary flex items-center justify-center border-[2.5px] transition-all duration-300 group-hover:shadow-lg"
                style={{ 
                  borderColor: homeColor,
                  boxShadow: `0 0 0 0 ${homeColor}00`,
                }}
              >
                <span className="text-xl font-bold text-foreground">
                  {homeTeam.charAt(0)}
                </span>
              </div>
              <div className="flex flex-col items-center gap-0.5">
                <span className="text-sm font-medium text-foreground text-center leading-tight">
                  {homeTeam}
                </span>
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  Home
                </span>
              </div>
            </div>

            {/* Score */}
            <div className="flex items-center gap-3">
              <span className="text-3xl font-bold text-foreground tabular-nums">
                {homeScore}
              </span>
              <span className="text-xl text-muted-foreground">-</span>
              <span className="text-3xl font-bold text-foreground tabular-nums">
                {awayScore}
              </span>
            </div>

            {/* Away Team */}
            <div className="flex-1 flex flex-col items-center gap-2">
              <div 
                className="w-12 h-12 rounded-full bg-secondary flex items-center justify-center border-[2.5px] transition-all duration-300 group-hover:shadow-lg"
                style={{ 
                  borderColor: awayColor,
                  boxShadow: `0 0 0 0 ${awayColor}00`,
                }}
              >
                <span className="text-xl font-bold text-foreground">
                  {awayTeam.charAt(0)}
                </span>
              </div>
              <div className="flex flex-col items-center gap-0.5">
                <span className="text-sm font-medium text-foreground text-center leading-tight">
                  {awayTeam}
                </span>
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  Away
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Combined Probability Bar */}
        <div className="px-4 pb-4">
          <div className="text-xs text-muted-foreground mb-3 uppercase tracking-wider">
            Win Probability
          </div>
          
          {/* Labels */}
          <div className="flex justify-between text-xs mb-1.5">
            <span className="text-[#00ff85] font-medium">Home {homeWinProb}%</span>
            <span className="text-muted-foreground font-medium">Draw {drawProb}%</span>
            <span className="text-[#04f5ff] font-medium">Away {awayWinProb}%</span>
          </div>
          
          {/* Combined Bar with Animation */}
          <div className="h-3 bg-secondary rounded-full overflow-hidden flex">
            <div
              className="h-full bg-[#00ff85] origin-left"
              style={{ 
                width: `${homeWinProb}%`,
                transform: isVisible ? 'scaleX(1)' : 'scaleX(0)',
                transition: 'transform 0.8s ease-out 0s',
              }}
            />
            <div
              className="h-full bg-muted-foreground/60 origin-left"
              style={{ 
                width: `${drawProb}%`,
                transform: isVisible ? 'scaleX(1)' : 'scaleX(0)',
                transition: 'transform 0.8s ease-out 0.1s',
              }}
            />
            <div
              className="h-full bg-[#04f5ff] origin-left"
              style={{ 
                width: `${awayWinProb}%`,
                transform: isVisible ? 'scaleX(1)' : 'scaleX(0)',
                transition: 'transform 0.8s ease-out 0.2s',
              }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
