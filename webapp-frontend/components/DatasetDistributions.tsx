'use client'

import { useState } from 'react'
import dynamic from 'next/dynamic'

const Plot = dynamic(
  () => import('react-plotly.js'),
  { 
    ssr: false,
    loading: () => <div className="h-[400px] flex items-center justify-center text-slate-500">Loading chart...</div>
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
  aresInitialized?: boolean
}

type MainTab = 'environment' | 'state-space' | 'action-space'
type EnvironmentSubTab = 'objects' | 'lighting' | 'materials'

export default function DatasetDistributions({ 
  datasetName, 
  isCurated = false,
  aresDistributions = [],
  aresInitialized = false
}: DatasetDistributionsProps) {
  const [activeMainTab, setActiveMainTab] = useState<MainTab>('environment')
  const [activeEnvSubTab, setActiveEnvSubTab] = useState<EnvironmentSubTab>('objects')
  const [activeStateSubTab, setActiveStateSubTab] = useState<'trajectory'>('trajectory')
  
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
      xaxis: { title: '', showgrid: true, gridcolor: '#2a2a2a', color: '#8a8a8a' },
      yaxis: { title: '', showgrid: true, gridcolor: '#2a2a2a', color: '#8a8a8a' },
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      font: { color: '#8a8a8a', size: 10 },
      height: 400,
      margin: { l: 40, r: 40, t: 20, b: 40 },
    },
    config: { displayModeBar: false },
  }

  return (
    <div>
      <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Dataset Distribution</h2>
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-3">
        {/* Main tabs */}
        <div className="flex gap-0 mb-3 border-b border-[#2a2a2a]">
          <button
            onClick={() => setActiveMainTab('environment')}
            className={`px-4 py-2 text-sm font-medium transition-colors relative ${
              activeMainTab === 'environment'
                ? 'text-[#d4d4d4]'
                : 'text-[#8a8a8a] hover:text-[#b4b4b4]'
            }`}
          >
            Environment
            {activeMainTab === 'environment' && (
              <span className="absolute bottom-0 left-0 right-0 h-px bg-[#154e72]" />
            )}
          </button>
          <button
            onClick={() => setActiveMainTab('state-space')}
            className={`px-4 py-2 text-sm font-medium transition-colors relative ${
              activeMainTab === 'state-space'
                ? 'text-[#d4d4d4]'
                : 'text-[#8a8a8a] hover:text-[#b4b4b4]'
            }`}
          >
            State-Space
            {activeMainTab === 'state-space' && (
              <span className="absolute bottom-0 left-0 right-0 h-px bg-[#154e72]" />
            )}
          </button>
          <button
            onClick={() => setActiveMainTab('action-space')}
            className={`px-4 py-2 text-sm font-medium transition-colors relative ${
              activeMainTab === 'action-space'
                ? 'text-[#d4d4d4]'
                : 'text-[#8a8a8a] hover:text-[#b4b4b4]'
            }`}
          >
            Action-Space
            {activeMainTab === 'action-space' && (
              <span className="absolute bottom-0 left-0 right-0 h-px bg-[#154e72]" />
            )}
          </button>
        </div>

        {/* Sub-tabs for Environment */}
        {activeMainTab === 'environment' && (
          <div className="flex gap-0 mb-3 border-b border-[#2a2a2a]">
            {objectsDist && (
              <button
                onClick={() => setActiveEnvSubTab('objects')}
                className={`px-3 py-1.5 text-xs font-medium transition-colors relative ${
                  activeEnvSubTab === 'objects'
                    ? 'text-[#d4d4d4]'
                    : 'text-[#8a8a8a] hover:text-[#b4b4b4]'
                }`}
              >
                Objects
                {activeEnvSubTab === 'objects' && (
                  <span className="absolute bottom-0 left-0 right-0 h-px bg-[#154e72]" />
                )}
              </button>
            )}
            {lightingDist && (
              <button
                onClick={() => setActiveEnvSubTab('lighting')}
                className={`px-3 py-1.5 text-xs font-medium transition-colors relative ${
                  activeEnvSubTab === 'lighting'
                    ? 'text-[#d4d4d4]'
                    : 'text-[#8a8a8a] hover:text-[#b4b4b4]'
                }`}
              >
                Lighting
                {activeEnvSubTab === 'lighting' && (
                  <span className="absolute bottom-0 left-0 right-0 h-px bg-[#154e72]" />
                )}
              </button>
            )}
            {materialsDist && (
              <button
                onClick={() => setActiveEnvSubTab('materials')}
                className={`px-3 py-1.5 text-xs font-medium transition-colors relative ${
                  activeEnvSubTab === 'materials'
                    ? 'text-[#d4d4d4]'
                    : 'text-[#8a8a8a] hover:text-[#b4b4b4]'
                }`}
              >
                Materials
                {activeEnvSubTab === 'materials' && (
                  <span className="absolute bottom-0 left-0 right-0 h-px bg-[#154e72]" />
                )}
              </button>
            )}
          </div>
        )}

        {/* Sub-tabs for State-space */}
        {activeMainTab === 'state-space' && (
          <div className="flex gap-0 mb-3 border-b border-[#2a2a2a]">
            <button
              onClick={() => setActiveStateSubTab('trajectory')}
              className={`px-3 py-1.5 text-xs font-medium transition-colors relative ${
                activeStateSubTab === 'trajectory'
                  ? 'text-[#d4d4d4]'
                  : 'text-[#8a8a8a] hover:text-[#b4b4b4]'
              }`}
            >
              Trajectory
              {activeStateSubTab === 'trajectory' && (
                <span className="absolute bottom-0 left-0 right-0 h-px bg-[#154e72]" />
              )}
            </button>
          </div>
        )}

        {/* Content area */}
        <div className="bg-[#1a1a1a] border border-[#2a2a2a]">
          {activeMainTab === 'environment' && (
            <>
              {aresInitialized && activeEnvSubTab === 'objects' && objectsDist && objectsDist.figure && (
                <Plot 
                  data={objectsDist.figure.data} 
                  layout={objectsDist.figure.layout} 
                  config={{ displayModeBar: false }} 
                  style={{ width: '100%', height: '400px' }}
                />
              )}
              {aresInitialized && activeEnvSubTab === 'lighting' && lightingDist && lightingDist.figure && (
                <Plot 
                  data={lightingDist.figure.data} 
                  layout={lightingDist.figure.layout} 
                  config={{ displayModeBar: false }} 
                  style={{ width: '100%', height: '400px' }}
                />
              )}
              {aresInitialized && activeEnvSubTab === 'materials' && materialsDist && materialsDist.figure && (
                <Plot 
                  data={materialsDist.figure.data} 
                  layout={materialsDist.figure.layout} 
                  config={{ displayModeBar: false }} 
                  style={{ width: '100%', height: '400px' }}
                />
              )}
              {!aresInitialized && (
                <div className="h-[400px] flex items-center justify-center text-[#8a8a8a] text-xs">
                  Loading distributions...
                </div>
              )}
              {aresInitialized && activeEnvSubTab === 'objects' && !objectsDist && (
                <div className="h-[400px] flex items-center justify-center text-[#8a8a8a] text-xs">
                  No Objects distribution available
                </div>
              )}
              {aresInitialized && activeEnvSubTab === 'lighting' && !lightingDist && (
                <div className="h-[400px] flex items-center justify-center text-[#8a8a8a] text-xs">
                  No Lighting distribution available
                </div>
              )}
              {aresInitialized && activeEnvSubTab === 'materials' && !materialsDist && (
                <div className="h-[400px] flex items-center justify-center text-[#8a8a8a] text-xs">
                  No Materials distribution available
                </div>
              )}
            </>
          )}
          {activeMainTab === 'state-space' && activeStateSubTab === 'trajectory' && (
            <Plot data={trajPlot.data} layout={trajPlot.layout} config={trajPlot.config} />
          )}
          {activeMainTab === 'action-space' && (
            <div className="h-[400px] flex items-center justify-center text-[#8a8a8a] text-xs">
              {/* Empty for action-space */}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

