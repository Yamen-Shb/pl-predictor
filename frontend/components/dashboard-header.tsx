import { Badge } from "@/components/ui/badge"
import { Trophy } from "lucide-react"

interface DashboardHeaderProps {
  gameweek: number
}

export function DashboardHeader({ gameweek }: DashboardHeaderProps) {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Trophy className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-xl md:text-2xl font-bold text-foreground tracking-tight">
                Premier League Predictions
              </h1>
              <p className="text-sm text-muted-foreground">
                AI-powered match predictions
              </p>
            </div>
          </div>
          <Badge
            variant="secondary"
            className="text-sm px-4 py-1.5 bg-primary/10 text-primary border-primary/20 border"
          >
            Gameweek {gameweek}
          </Badge>
        </div>
      </div>
    </header>
  )
}
