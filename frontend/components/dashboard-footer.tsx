import { RefreshCw } from "lucide-react"

interface DashboardFooterProps {
  lastUpdated: string
}

export function DashboardFooter({ lastUpdated }: DashboardFooterProps) {
  return (
    <footer className="border-t border-border bg-card/30 mt-auto">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between flex-wrap gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Last updated: {lastUpdated}</span>
          </div>
          
          <div className="flex items-center gap-4">
            <span>Premier League Predictions Dashboard</span>
            <div className="flex items-center gap-3">
              <a 
                href="https://www.linkedin.com/in/yamen-shehab/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-muted-foreground hover:text-blue-500 transition-colors underline-offset-4 hover:underline"
              >
                LinkedIn
              </a>
              <span>•</span>
              <a 
                href="https://github.com/Yamen-Shb" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-muted-foreground hover:text-foreground transition-colors underline-offset-4 hover:underline"
              >
                GitHub
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}