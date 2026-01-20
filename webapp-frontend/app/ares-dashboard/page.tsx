'use client'

import { useState, useEffect } from 'react'
import { Loader2, Filter, BarChart3, Download, Table } from 'lucide-react'
import { cn } from '@/lib/utils'
import dynamic from 'next/dynamic'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

interface AresState {
  total_rows: number
  total_statistics: any
  annotation_statistics: any
  has_embeddings: boolean
  has_annotations: boolean
}

interface Visualization {
  title: string
  figure: any
}

interface FilteredData {
  filtered_count: number
  total_count: number
  active_filters: any
  data: any[]
}

export default function AresDashboard() {
  const [initialized, setInitialized] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [state, setState] = useState<AresState | null>(null)
  
  // Filter state
  const [filterMetadata, setFilterMetadata] = useState<any[]>([])
  const [structuredFilters, setStructuredFilters] = useState<any>({})
  const [filteredData, setFilteredData] = useState<FilteredData | null>(null)
  const [dataSample, setDataSample] = useState<any[]>([])
  const [showFilters, setShowFilters] = useState(true) // Show filters by default
  const [loadingMetadata, setLoadingMetadata] = useState(false)
  
  // Embedding filters state
  const [embeddingData, setEmbeddingData] = useState<Record<string, any>>({})
  const [embeddingSelections, setEmbeddingSelections] = useState<Record<string, string[]>>({})
  const [activeEmbeddingTab, setActiveEmbeddingTab] = useState<string | null>(null)
  const [loadingEmbeddings, setLoadingEmbeddings] = useState(false)
  
  // Visualization state
  const [distributions, setDistributions] = useState<Visualization[]>([])
  const [activeDistributionTab, setActiveDistributionTab] = useState<number>(0)
  const [timeSeries, setTimeSeries] = useState<Visualization[]>([])
  
  // Hero and robot array state
  const [selectedRowId, setSelectedRowId] = useState<string | null>(null)
  const [heroData, setHeroData] = useState<any | null>(null)
  const [robotArrayPlots, setRobotArrayPlots] = useState<Visualization[]>([])

  // ARES initialization removed - data loads lazily when needed

  // Load filter metadata
  useEffect(() => {
    if (!initialized) return

    const loadMetadata = async () => {
      setLoadingMetadata(true)
      try {
        const response = await fetch('/api/ares/filters/metadata')
        if (response.ok) {
          const data = await response.json()
          console.log('Filter metadata loaded:', data.columns?.length, 'columns')
          setFilterMetadata(data.columns || [])
        } else {
          const errorText = await response.text()
          console.error('Error loading filter metadata:', response.status, errorText)
        }
      } catch (err) {
        console.error('Error loading filter metadata:', err)
      } finally {
        setLoadingMetadata(false)
      }
    }

    loadMetadata()
  }, [initialized])

  // Load embedding data
  useEffect(() => {
    if (!initialized || !state?.has_embeddings) return

    const loadEmbeddings = async () => {
      setLoadingEmbeddings(true)
      try {
        // Load available embedding types
        const embeddingKeys = ['task_language_instruction', 'description_estimate']
        const embeddingDataMap: Record<string, any> = {}
        
        for (const key of embeddingKeys) {
          try {
            const response = await fetch(`/api/ares/embeddings/${key}`)
            if (response.ok) {
              const data = await response.json()
              embeddingDataMap[key] = data
              console.log(`Loaded embedding data for ${key}:`, data.point_count, 'points')
              if (!activeEmbeddingTab) {
                setActiveEmbeddingTab(key)
              }
            } else {
              const errorText = await response.text()
              console.error(`Failed to load embedding ${key}:`, response.status, errorText)
            }
          } catch (err) {
            console.error(`Error loading embedding ${key}:`, err)
          }
        }
        
        console.log('Embedding data loaded:', Object.keys(embeddingDataMap).length, 'embeddings')
        setEmbeddingData(embeddingDataMap)
      } catch (err) {
        console.error('Error loading embeddings:', err)
      } finally {
        setLoadingEmbeddings(false)
      }
    }

    loadEmbeddings()
  }, [initialized, state?.has_embeddings])

  // Apply structured filters and load visualizations
  useEffect(() => {
    if (!initialized) return

    const applyFiltersAndLoad = async () => {
      try {
        // Apply structured filters
        const filterResponse = await fetch('/api/ares/filters/structured', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(structuredFilters),
        })
        
        if (filterResponse.ok) {
          const filterData = await filterResponse.json()
          setFilteredData(filterData)
          
          // Get a random sample of 5 rows
          if (filterData.data && filterData.data.length > 0) {
            const sample = filterData.data
              .sort(() => Math.random() - 0.5)
              .slice(0, 5)
            setDataSample(sample)
          }
        }

        // Distributions are loaded from the dataset-specific pages with proper parameters
        // Don't load them here to avoid duplicate generation

        // Load time series
        const timeResponse = await fetch('/api/ares/time-series')
        if (timeResponse.ok) {
          const timeData = await timeResponse.json()
          setTimeSeries(timeData.visualizations || [])
        }
      } catch (err: any) {
        console.error('Error loading data:', err)
      }
    }

    applyFiltersAndLoad()
  }, [initialized, structuredFilters])

  // Load hero data when row is selected
  useEffect(() => {
    if (!selectedRowId) {
      setHeroData(null)
      setRobotArrayPlots([])
      return
    }

    const loadHeroData = async () => {
      try {
        const [heroResponse, robotResponse] = await Promise.all([
          fetch(`/api/ares/hero/${selectedRowId}`),
          fetch(`/api/ares/robot-array/${selectedRowId}`),
        ])
        
        if (heroResponse.ok) {
          const data = await heroResponse.json()
          setHeroData(data)
        }
        
        if (robotResponse.ok) {
          const data = await robotResponse.json()
          setRobotArrayPlots(data.visualizations || [])
        }
      } catch (err: any) {
        console.error('Error loading hero data:', err)
      }
    }

    loadHeroData()
  }, [selectedRowId])

  if (loading && !initialized) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-[#1a1a1a]">
        <Loader2 className="w-8 h-8 animate-spin text-[#666666]" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-[#1a1a1a] text-[#d4d4d4] p-8">
        <div className="bg-[#2a1a1a] border border-[#3a2a2a] p-4">
          <p className="text-[#cc6666]">Error: {error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[#1a1a1a] text-[#d4d4d4]">
      {/* Header */}
      <header className="border-b border-[#2a2a2a] bg-[#222222] sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-between h-10">
            <h1 className="text-sm font-medium tracking-wide text-white">
              6d labs - ARES Dashboard
            </h1>
            {state && (
              <div className="text-xs text-[#8a8a8a]">
                {state.total_rows} rows loaded
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* 1. State Info Section - HIDDEN */}
        {false && state && (
          <div className="bg-[#222222] border border-[#2a2a2a] p-4">
            <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">System State</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div>
                <div className="text-[#8a8a8a]">Total Rows</div>
                <div className="text-[#d4d4d4] font-medium">{state.total_rows}</div>
              </div>
              <div>
                <div className="text-[#8a8a8a]">Has Embeddings</div>
                <div className="text-[#d4d4d4] font-medium">{state.has_embeddings ? 'Yes' : 'No'}</div>
              </div>
              <div>
                <div className="text-[#8a8a8a]">Has Annotations</div>
                <div className="text-[#d4d4d4] font-medium">{state.has_annotations ? 'Yes' : 'No'}</div>
              </div>
            </div>
          </div>
        )}

        {/* Divider - HIDDEN */}
        {false && <div className="border-t border-[#2a2a2a]" />}

        {/* 2. Structured Data Filters Section - HIDDEN */}
        {false && <div className="bg-[#222222] border border-[#2a2a2a] p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xs font-medium text-[#d4d4d4] flex items-center gap-2">
              <Filter className="w-4 h-4" />
              Structured Data Filters
            </h2>
            <button
              onClick={() => setStructuredFilters({})}
              className="px-3 py-1 text-xs bg-[#4b6671] text-white hover:bg-[#3d5560] transition-colors"
            >
              Reset Filters
            </button>
          </div>
          {filteredData ? (
            <>
              <div className="text-xs text-[#8a8a8a] mb-4">
                Selected {filteredData.filtered_count} rows out of {filteredData.total_count} total via structured data filters
              </div>
              {filteredData.filtered_count === 0 && (
                <div className="text-xs text-[#cc6666] mb-4">
                  No data matches the structured filters!
                </div>
              )}
            </>
          ) : (
            <div className="text-xs text-[#8a8a8a] mb-4">
              No filters applied yet. Use the filters below to filter the data.
            </div>
          )}
          {/* Filter Toggle */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="mb-4 px-3 py-1 text-xs bg-[#2a2a2a] text-[#d4d4d4] hover:bg-[#3a3a3a] transition-colors border border-[#2a2a2a]"
          >
            {showFilters ? 'Hide Filters' : 'Show Filters'}
          </button>

          {/* Loading state */}
          {loadingMetadata && (
            <div className="text-xs text-[#8a8a8a] mb-4">Loading filter metadata...</div>
          )}

          {/* No metadata message */}
          {!loadingMetadata && filterMetadata.length === 0 && (
            <div className="text-xs text-[#8a8a8a] mb-4">
              No filterable columns found. This might indicate an issue loading the data.
            </div>
          )}

          {/* Filter UI */}
          {showFilters && filterMetadata.length > 0 && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filterMetadata.map((colMeta: any) => {
                  if (colMeta.viz_type === 'histogram' && colMeta.min !== undefined && colMeta.min !== null && colMeta.max !== undefined && colMeta.max !== null) {
                    // Numeric slider
                    const currentRange = structuredFilters[colMeta.column]?.range || [colMeta.min, colMeta.max]
                    const step = (colMeta.max - colMeta.min) / 100
                    return (
                      <div key={colMeta.column} className="space-y-2">
                        <label className="text-xs text-[#d4d4d4] font-medium">
                          {colMeta.column.replace(/_/g, ' ')}
                        </label>
                        <div className="flex gap-2 items-center">
                          <input
                            type="number"
                            value={currentRange[0]}
                            onChange={(e) => {
                              const newFilters = { ...structuredFilters }
                              if (!newFilters[colMeta.column]) {
                                newFilters[colMeta.column] = { range: [colMeta.min, colMeta.max] }
                              }
                              newFilters[colMeta.column].range[0] = parseFloat(e.target.value) || colMeta.min
                              setStructuredFilters(newFilters)
                            }}
                            min={colMeta.min}
                            max={colMeta.max}
                            step={step}
                            className="w-20 px-2 py-1 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4]"
                          />
                          <input
                            type="range"
                            min={colMeta.min}
                            max={colMeta.max}
                            step={step}
                            value={currentRange[1]}
                            onChange={(e) => {
                              const newFilters = { ...structuredFilters }
                              if (!newFilters[colMeta.column]) {
                                newFilters[colMeta.column] = { range: [colMeta.min, colMeta.max] }
                              }
                              newFilters[colMeta.column].range[1] = parseFloat(e.target.value)
                              setStructuredFilters(newFilters)
                            }}
                            className="flex-1"
                          />
                          <input
                            type="number"
                            value={currentRange[1]}
                            onChange={(e) => {
                              const newFilters = { ...structuredFilters }
                              if (!newFilters[colMeta.column]) {
                                newFilters[colMeta.column] = { range: [colMeta.min, colMeta.max] }
                              }
                              newFilters[colMeta.column].range[1] = parseFloat(e.target.value) || colMeta.max
                              setStructuredFilters(newFilters)
                            }}
                            min={colMeta.min}
                            max={colMeta.max}
                            step={step}
                            className="w-20 px-2 py-1 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4]"
                          />
                        </div>
                        {colMeta.has_nan && (
                          <label className="flex items-center gap-2 text-xs text-[#8a8a8a]">
                            <input
                              type="checkbox"
                              checked={structuredFilters[colMeta.column]?.include_none !== false}
                              onChange={(e) => {
                                const newFilters = { ...structuredFilters }
                                if (!newFilters[colMeta.column]) {
                                  newFilters[colMeta.column] = { range: [colMeta.min, colMeta.max] }
                                }
                                newFilters[colMeta.column].include_none = e.target.checked
                                setStructuredFilters(newFilters)
                              }}
                              className="w-3 h-3"
                            />
                            Include None values
                          </label>
                        )}
                      </div>
                    )
                  } else if (colMeta.viz_type === 'bar' && colMeta.options) {
                    // Categorical multi-select
                    const selectedValues = structuredFilters[colMeta.column]?.values || colMeta.options
                    return (
                      <div key={colMeta.column} className="space-y-2">
                        <label className="text-xs text-[#d4d4d4] font-medium">
                          {colMeta.column.replace(/_/g, ' ')}
                        </label>
                        <select
                          multiple
                          value={selectedValues}
                          onChange={(e) => {
                            const selected = Array.from(e.target.selectedOptions, option => option.value)
                            const newFilters = { ...structuredFilters }
                            newFilters[colMeta.column] = { values: selected }
                            setStructuredFilters(newFilters)
                          }}
                          className="w-full px-2 py-1 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] min-h-[80px]"
                          size={Math.min(colMeta.options.length, 5)}
                        >
                          {colMeta.options.map((opt: string) => (
                            <option key={opt} value={opt} className="bg-[#1a1a1a] text-[#d4d4d4]">
                              {opt}
                            </option>
                          ))}
                        </select>
                        <div className="text-xs text-[#8a8a8a]">
                          {selectedValues.length} selected (hold Ctrl/Cmd to select multiple)
                        </div>
                      </div>
                    )
                  }
                  return null
                })}
              </div>
              <button
                onClick={() => {
                  // Trigger filter application
                  setStructuredFilters({ ...structuredFilters })
                }}
                className="px-4 py-2 text-xs bg-[#4b6671] text-white hover:bg-[#3d5560] transition-colors"
              >
                Apply Filters
              </button>
            </div>
          )}
        </div>}

        {/* Divider - HIDDEN */}
        {false && <div className="border-t border-[#2a2a2a]" />}

        {/* 3. Embedding Data Filters Section */}
        {state?.has_embeddings && (
          <>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6">
              <h2 className="text-xs font-medium mb-4 text-[#d4d4d4]">Embedding Data Filters</h2>
              
              {Object.keys(embeddingData).length === 0 ? (
                <div className="text-xs text-[#8a8a8a] mb-4">
                  {loadingEmbeddings ? 'Loading embedding data...' : 'No embedding data available. Embeddings may not have been generated during initialization. Check the browser console for details.'}
                </div>
              ) : (
                <>
                  {/* Embedding type tabs */}
                  <div className="border-b border-[#2a2a2a] mb-4">
                    <div className="flex gap-1">
                      {Object.keys(embeddingData).map((key) => (
                        <button
                          key={key}
                          onClick={() => setActiveEmbeddingTab(key)}
                          className={cn(
                            "px-4 py-2 text-xs font-medium transition-colors",
                            activeEmbeddingTab === key
                              ? "text-[#d4d4d4] border-b-2 border-[#4b6671]"
                              : "text-[#8a8a8a] hover:text-[#d4d4d4]"
                          )}
                        >
                          {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Active embedding plot */}
                  {activeEmbeddingTab && embeddingData[activeEmbeddingTab] && (
                    <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4">
                      <div className="mb-2 text-xs text-[#8a8a8a]">
                        {embeddingData[activeEmbeddingTab].point_count} points. 
                        Use box select, lasso select, or click points to filter data.
                      </div>
                      {embeddingData[activeEmbeddingTab].figure && (
                        <Plot
                          data={embeddingData[activeEmbeddingTab].figure.data}
                          layout={{
                            ...embeddingData[activeEmbeddingTab].figure.layout,
                            dragmode: 'lasso', // Enable lasso selection
                            selectdirection: 'diagonal', // Enable box selection
                          }}
                          config={{
                            displayModeBar: false,
                            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                          }}
                          style={{ width: '100%', height: '500px' }}
                          onSelected={(data: any) => {
                            if (data && data.points) {
                              // Extract selected point indices
                              const selectedIndices = data.points.map((p: any) => p.pointNumber)
                              // Get IDs from the customdata
                              const selectedIds = data.points
                                .map((p: any) => p.customdata?.[1] || p.customdata?.[0])
                                .filter((id: any) => id !== undefined)
                              
                              setEmbeddingSelections({
                                ...embeddingSelections,
                                [activeEmbeddingTab]: selectedIds,
                              })
                              
                              console.log(`Selected ${selectedIds.length} points in ${activeEmbeddingTab}`)
                            }
                          }}
                          onDeselect={() => {
                            setEmbeddingSelections({
                              ...embeddingSelections,
                              [activeEmbeddingTab]: [],
                            })
                          }}
                        />
                      )}
                      {embeddingSelections[activeEmbeddingTab] && embeddingSelections[activeEmbeddingTab].length > 0 && (
                        <div className="mt-2 text-xs text-[#8a8a8a]">
                          {embeddingSelections[activeEmbeddingTab].length} points selected
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
            <div className="border-t border-[#2a2a2a]" />
          </>
        )}

        {/* 4. Data Sample Section - HIDDEN */}
        {false && dataSample.length > 0 && (
          <>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6">
              <h2 className="text-xs font-medium mb-4 text-[#d4d4d4] flex items-center gap-2">
                <Table className="w-4 h-4" />
                Data Sample
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-[#2a2a2a]">
                      {Object.keys(dataSample[0] || {}).slice(0, 10).map((key) => (
                        <th key={key} className="text-left p-2 text-[#8a8a8a] font-medium">
                          {key.replace(/_/g, ' ')}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {dataSample.map((row, idx) => (
                      <tr
                        key={idx}
                        className={cn(
                          "border-b border-[#2a2a2a] hover:bg-[#1a1a1a] cursor-pointer",
                          selectedRowId === row.id && "bg-[#2a3a3a]"
                        )}
                        onClick={() => setSelectedRowId(row.id)}
                      >
                        {Object.values(row).slice(0, 10).map((val: any, i) => (
                          <td key={i} className="p-2 text-[#d4d4d4]">
                            {typeof val === 'object' ? JSON.stringify(val).slice(0, 50) : String(val).slice(0, 50)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="border-t border-[#2a2a2a]" />
          </>
        )}

        {/* 5. Data Distributions Section */}
        {distributions.length > 0 && (
          <>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6">
              <h2 className="text-xs font-medium mb-4 text-[#d4d4d4] flex items-center gap-2">
                <BarChart3 className="w-4 h-4" />
                Data Distributions
              </h2>
              
              {/* Tabs */}
              <div className="border-b border-[#2a2a2a] mb-4">
                <div className="flex gap-1 overflow-x-auto">
                  {distributions.map((viz, idx) => (
                    <button
                      key={idx}
                      onClick={() => setActiveDistributionTab(idx)}
                      className={cn(
                        "px-4 py-2 text-xs font-medium transition-colors whitespace-nowrap",
                        activeDistributionTab === idx
                          ? "text-[#d4d4d4] border-b-2 border-[#4b6671]"
                          : "text-[#8a8a8a] hover:text-[#d4d4d4]"
                      )}
                    >
                      {viz.title}
                    </button>
                  ))}
                </div>
              </div>

              {/* Active Tab Content */}
              {distributions[activeDistributionTab] && (
                <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4">
                  {distributions[activeDistributionTab].figure && (
                    <Plot
                      data={distributions[activeDistributionTab].figure.data}
                      layout={distributions[activeDistributionTab].figure.layout}
                      config={{ displayModeBar: false }}
                      style={{ width: '100%', height: '400px' }}
                    />
                  )}
                </div>
              )}
            </div>
            <div className="border-t border-[#2a2a2a]" />
          </>
        )}

        {/* 6. Time Series Trends Section - HIDDEN */}
        {false && timeSeries.length > 0 && (
          <>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6">
              <h2 className="text-xs font-medium mb-4 text-[#d4d4d4]">Time Series Trends</h2>
              <div className="space-y-4">
                {timeSeries.map((viz, idx) => (
                  <div key={idx} className="bg-[#1a1a1a] border border-[#2a2a2a] p-4">
                    <h3 className="text-xs font-medium mb-2 text-[#d4d4d4]">{viz.title}</h3>
                    {viz.figure && (
                      <Plot
                        data={viz.figure.data}
                        layout={viz.figure.layout}
                        config={{ displayModeBar: false }}
                        style={{ width: '100%', height: '400px' }}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
            <div className="border-t border-[#2a2a2a]" />
          </>
        )}

        {/* 7. Hero Display Section - HIDDEN */}
        {false && heroData && (
          <>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6">
              <h2 className="text-xs font-medium mb-4 text-[#d4d4d4]">Selected Row Details</h2>
              <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4">
                <pre className="text-xs text-[#8a8a8a] overflow-auto max-h-96">
                  {JSON.stringify(heroData.selected_row, null, 2)}
                </pre>
              </div>
            </div>
            <div className="border-t border-[#2a2a2a]" />
          </>
        )}

        {/* 8. Robot Array Plots Section - HIDDEN */}
        {false && robotArrayPlots.length > 0 && (
          <>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6">
              <h2 className="text-xs font-medium mb-4 text-[#d4d4d4]">Robot Array Plots</h2>
              <div className="space-y-4">
                {robotArrayPlots.map((viz, idx) => (
                  <div key={idx} className="bg-[#1a1a1a] border border-[#2a2a2a] p-4">
                    <h3 className="text-xs font-medium mb-2 text-[#d4d4d4]">{viz.title}</h3>
                    {viz.figure && (
                      <Plot
                        data={viz.figure.data}
                        layout={viz.figure.layout}
                        config={{ displayModeBar: false }}
                        style={{ width: '100%', height: '400px' }}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
            <div className="border-t border-[#2a2a2a]" />
          </>
        )}

        {/* 9. Export Section - HIDDEN */}
        {false && <div className="bg-[#222222] border border-[#2a2a2a] p-6">
          <h2 className="text-xs font-medium mb-4 text-[#d4d4d4] flex items-center gap-2">
            <Download className="w-4 h-4" />
            Export Data
          </h2>
          <div className="flex gap-2">
            <button
              onClick={async () => {
                try {
                  const response = await fetch('/api/ares/export?format=csv', {
                    method: 'POST'
                  })
                  if (response.ok) {
                    const data = await response.json()
                    const blob = new Blob([data.content], { type: 'text/csv' })
                    const url = window.URL.createObjectURL(blob)
                    const a = document.createElement('a')
                    a.href = url
                    a.download = data.filename
                    a.click()
                  }
                } catch (err) {
                  console.error('Export error:', err)
                }
              }}
              className="px-4 py-2 text-xs bg-[#4b6671] text-white hover:bg-[#3d5560] transition-colors"
            >
              Export CSV
            </button>
            <button
              onClick={async () => {
                try {
                  const response = await fetch('/api/ares/export?format=json', {
                    method: 'POST'
                  })
                  if (response.ok) {
                    const data = await response.json()
                    const blob = new Blob([data.content], { type: 'application/json' })
                    const url = window.URL.createObjectURL(blob)
                    const a = document.createElement('a')
                    a.href = url
                    a.download = data.filename
                    a.click()
                  }
                } catch (err) {
                  console.error('Export error:', err)
                }
              }}
              className="px-4 py-2 text-xs bg-[#4b6671] text-white hover:bg-[#3d5560] transition-colors"
            >
              Export JSON
            </button>
          </div>
        </div>}
      </main>
    </div>
  )
}
