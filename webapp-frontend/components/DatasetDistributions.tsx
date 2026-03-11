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

type EnvironmentSubTab = 'objects' | 'lighting' | 'materials'

export default function DatasetDistributions({
  datasetName,
  isCurated = false,
  aresDistributions = [],
  loading = false
}: DatasetDistributionsProps) {
  const [activeEnvSubTab, setActiveEnvSubTab] = useState<EnvironmentSubTab>('objects')

  // Auto-select first available tab when distributions load
  useEffect(() => {
    if (aresDistributions.length > 0) {
      const currentTabExists = aresDistributions.some(
        dist => dist.title.toLowerCase().replace(/\s+/g, '-') === activeEnvSubTab
      )
      if (!currentTabExists) {
        const firstTabKey = aresDistributions[0].title.toLowerCase().replace(/\s+/g, '-')
        setActiveEnvSubTab(firstTabKey as EnvironmentSubTab)
      }
    }
  }, [aresDistributions, activeEnvSubTab])

  return (
    <div>
      <h2 className="text-xs font-medium mb-3 text-white">Dataset Distribution</h2>
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-3">
        {/* Sub-tabs */}
        {aresDistributions.length > 0 && (
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

        {/* Content area */}
        <div className="bg-[#1a1a1a] border border-white/10">
          {loading ? (
            <div className="h-[400px] flex items-center justify-center text-white/30 text-xs">
              Loading distributions...
            </div>
          ) : aresDistributions.length > 0 ? (
            <div className="p-4">
              {aresDistributions.map((dist, idx) => {
                const tabKey = dist.title.toLowerCase().replace(/\s+/g, '-')
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
        </div>
      </div>
    </div>
  )
}
