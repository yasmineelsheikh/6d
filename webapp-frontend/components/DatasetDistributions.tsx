'use client'

import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'

const Plot = dynamic(
  () => import('react-plotly.js'),
  {
    ssr: false,
    loading: () => <div className="h-[400px] flex items-center justify-center text-white/30">Loading chart...</div>
  }
)

interface Visualization {
  title: string
  figure: any
}

interface DatasetDistributionsProps {
  datasetName: string
  isCurated?: boolean
  aresDistributions?: Visualization[]
  loading?: boolean
}

type MainTab = 'environment' | 'state-space' | 'action-space'
type EnvironmentSubTab = 'objects' | 'lighting' | 'materials'

export default function DatasetDistributions({
  datasetName,
  isCurated = false,
  aresDistributions = [],
  loading = false
}: DatasetDistributionsProps) {
  const [activeMainTab, setActiveMainTab] = useState<MainTab>('environment')
  const [activeEnvSubTab, setActiveEnvSubTab] = useState<EnvironmentSubTab>('objects')
  const [activeStateSubTab, setActiveStateSubTab] = useState<'trajectory'>('trajectory')

  // Auto-select first available tab when distributions load
  useEffect(() => {
    if (aresDistributions.length > 0 && activeMainTab === 'environment') {
      // Get the first distribution's tab key
      const firstTabKey = aresDistributions[0].title.toLowerCase().replace(/\s+/g, '-')
      // Only update if the current tab doesn't match any available distribution
      const currentTabExists = aresDistributions.some(
        dist => dist.title.toLowerCase().replace(/\s+/g, '-') === activeEnvSubTab
      )
      if (!currentTabExists) {
        setActiveEnvSubTab(firstTabKey as EnvironmentSubTab)
      }
    }
  }, [aresDistributions, activeMainTab, activeEnvSubTab])

  // Map ares distribution titles to sub-tabs
  const getDistributionByTitle = (title: string): Visualization | undefined => {
    return aresDistributions.find(viz => viz.title.toLowerCase() === title.toLowerCase())
  }

  const objectsDist = getDistributionByTitle('Objects')
  const lightingDist = getDistributionByTitle('Lighting')
  const materialsDist = getDistributionByTitle('Materials')

  const generateTrajectoryData = () => {
    const traces: Array<{
      x: number[]
      y: number[]
      line: { width: number }
      opacity: number
      showlegend: boolean
      hoverinfo: 'skip'
    }> = []
    for (let i = 0; i < 15; i++) {
      const length = Math.floor(Math.random() * 60) + 20
      const x = Array.from({ length }, (_, i) => i)
      const y = x.map((_, idx) => {
        return (
          Math.sin(idx / 5) * 2 +
          Array.from({ length: idx + 1 }, () => Math.random() * 0.5 - 0.25).reduce((a, b) => a + b, 0)
        )
      })
      traces.push({
        x,
        y,
        line: { width: 1.5 },
        opacity: 0.6,
        showlegend: false,
        hoverinfo: 'skip' as const,
      })
    }
    return traces
  }

  const trajData = generateTrajectoryData()

  const trajPlot = {
    data: trajData.map(trace => ({
      ...trace,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { ...trace.line, color: '#154e72' },
    })),
    layout: {
      title: '',
      xaxis: { title: '', showgrid: true, gridcolor: 'rgba(255,255,255,0.05)', color: 'rgba(255,255,255,0.6)' },
      yaxis: { title: '', showgrid: true, gridcolor: 'rgba(255,255,255,0.05)', color: 'rgba(255,255,255,0.6)' },
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      font: { color: 'rgba(255,255,255,0.6)', size: 11 },
      height: 400,
      margin: { l: 40, r: 40, t: 20, b: 40 },
    },
    config: { displayModeBar: false },
  }

  return (
    <div>
      <h2 className="text-xs font-medium mb-3 text-white">Dataset Distribution</h2>
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-3">
        {/* Main tabs */}
        <div className="flex gap-0 mb-3 border-b border-white/10">
          <button
            onClick={() => setActiveMainTab('environment')}
            className={`px-4 py-2 text-sm font-medium transition-colors relative ${activeMainTab === 'environment'
                ? 'text-white'
                : 'text-white/40 hover:text-white'
              }`}
          >
            Environment
            {activeMainTab === 'environment' && (
              <span className="absolute bottom-0 left-0 right-0 h-px bg-[#4b6671]" />
            )}
          </button>
          <button
            type="button"
            className="px-4 py-2 text-sm font-medium text-white/20 cursor-not-allowed"
          >
            State-Space
          </button>
          <button
            type="button"
            className="px-4 py-2 text-sm font-medium text-white/20 cursor-not-allowed"
          >
            Action-Space
          </button>
        </div>

        {/* Sub-tabs for Environment - show all available distributions */}
        {activeMainTab === 'environment' && aresDistributions.length > 0 && (
          <div className="flex gap-0 mb-3 border-b border-white/10">
            {aresDistributions.map((dist, idx) => {
              const tabKey = dist.title.toLowerCase().replace(/\s+/g, '-')
              return (
                <button
                  key={idx}
                  onClick={() => setActiveEnvSubTab(tabKey as EnvironmentSubTab)}
                  className={`px-3 py-1.5 text-xs font-medium transition-colors relative ${activeEnvSubTab === tabKey
                      ? 'text-white'
                      : 'text-white/40 hover:text-white'
                    }`}
                >
                  {dist.title}
                  {activeEnvSubTab === tabKey && (
                    <span className="absolute bottom-0 left-0 right-0 h-px bg-[#4b6671]" />
                  )}
                </button>
              )
            })}
          </div>
        )}

        {/* Sub-tabs for State-space (disabled / coming soon) */}
        {activeMainTab === 'state-space' && (
          <div className="mb-3 text-[10px] text-white/30 italic">
            State-space visualizations coming soon.
          </div>
        )}

        {/* Content area */}
        <div className="bg-[#1a1a1a] border border-white/10">
          {activeMainTab === 'environment' && (
            <>
              {loading ? (
                <div className="h-[400px] flex items-center justify-center text-white/30 text-xs">
                  Loading distributions...
                </div>
              ) : aresDistributions.length > 0 ? (
                <div className="p-4">
                  {aresDistributions.map((dist, idx) => {
                    const tabKey = dist.title.toLowerCase().replace(/\s+/g, '-')
                    // Show plot if it matches active tab, or show all if no tab is selected
                    const shouldShow = !activeEnvSubTab || activeEnvSubTab === tabKey
                    return shouldShow && dist.figure ? (
                      <Plot
                        key={idx}
                        data={dist.figure.data}
                        layout={dist.figure.layout}
                        config={{ displayModeBar: false }}
                        style={{ width: '100%', height: '400px' }}
                      />
                    ) : null
                  })}
                </div>
              ) : (
                <div className="h-[400px] flex items-center justify-center text-white/30 text-xs">
                  No distributions available
                </div>
              )}
            </>
          )}
          {activeMainTab === 'state-space' && (
            <div className="h-[400px] flex items-center justify-center text-white/30 text-xs italic">
              State-space plots are not yet available.
            </div>
          )}
          {activeMainTab === 'action-space' && (
            <div className="h-[400px] flex items-center justify-center text-white/30 text-xs italic">
              Action-space plots are not yet available.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}




