'use client'

import { useState, useEffect } from 'react'
import { Upload, Play, Download, Loader2, CheckCircle2, XCircle, Filter } from 'lucide-react'
import DatasetUpload from '@/components/DatasetUpload'
import DatasetOverview from '@/components/DatasetOverview'
import DatasetDistributions from '@/components/DatasetDistributions'
import EpisodePreview from '@/components/EpisodePreview'
import AugmentationPanel from '@/components/AugmentationPanel'
import OptimizationPanel from '@/components/OptimizationPanel'
import TestingPanel from '@/components/TestingPanel'
import { cn } from '@/lib/utils'
import dynamic from 'next/dynamic'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

interface DatasetInfo {
  dataset_name: string
  total_episodes: number
  robot_type: string
}

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

export default function Home() {
  const [currentDataset, setCurrentDataset] = useState<string | null>(null)
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [datasetData, setDatasetData] = useState<any[]>([])
  const [curatedData, setCuratedData] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [curationTab, setCurationTab] = useState<'automated' | 'manual'>('automated')
  
  // ARES state
  const [aresInitialized, setAresInitialized] = useState(false)
  const [aresLoading, setAresLoading] = useState(false)
  const [aresError, setAresError] = useState<string | null>(null)
  const [aresState, setAresState] = useState<AresState | null>(null)
  
  // Embedding filters state
  const [embeddingData, setEmbeddingData] = useState<Record<string, any>>({})
  const [embeddingSelections, setEmbeddingSelections] = useState<Record<string, string[]>>({})
  const [activeEmbeddingTab, setActiveEmbeddingTab] = useState<string | null>(null)
  const [loadingEmbeddings, setLoadingEmbeddings] = useState(false)
  
  // Distributions state
  const [aresDistributions, setAresDistributions] = useState<Visualization[]>([])

  const handleDatasetLoaded = async (datasetName: string) => {
    setCurrentDataset(datasetName)
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`/api/datasets/${datasetName}/info`)
      if (!response.ok) throw new Error('Failed to load dataset info')
      const info = await response.json()
      setDatasetInfo(info)
      
      const dataResponse = await fetch(`/api/datasets/${datasetName}/data`)
      if (!dataResponse.ok) throw new Error('Failed to load dataset data')
      const data = await dataResponse.json()
      setDatasetData(data.data || [])
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleAugmentationComplete = async (datasetName: string) => {
    // Reload dataset data for curated view
    try {
      const dataResponse = await fetch(`/api/datasets/${datasetName}/data`)
      if (!dataResponse.ok) throw new Error('Failed to load curated data')
      const data = await dataResponse.json()
      setCuratedData(data.data || [])
    } catch (err: any) {
      setError(err.message)
    }
  }

  // Initialize ARES data
  useEffect(() => {
    const init = async () => {
      setAresLoading(true)
      try {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 60000)
        
        const response = await fetch('/api/ares/initialize', {
          method: 'POST',
          signal: controller.signal,
        })
        clearTimeout(timeoutId)
        
        if (!response.ok) {
          let errorMessage = `Failed to initialize ARES: ${response.status} ${response.statusText}`
          try {
            const text = await response.text()
            try {
              const errorData = JSON.parse(text)
              errorMessage = errorData.detail || errorData.message || text || errorMessage
            } catch {
              errorMessage = text || errorMessage
            }
          } catch (e) {
            // Keep default error message
          }
          throw new Error(errorMessage)
        }
        
        const initData = await response.json()
        const stateResponse = await fetch('/api/ares/state')
        if (!stateResponse.ok) {
          throw new Error('Failed to get state')
        }
        const stateData = await stateResponse.json()
        setAresState(stateData)
        setAresInitialized(true)
      } catch (err: any) {
        let errorMsg = err.message || err.toString() || 'Unknown error occurred'
        if (err.name === 'AbortError') {
          errorMsg = 'Initialization timed out after 60 seconds.'
        }
        setAresError(errorMsg)
      } finally {
        setAresLoading(false)
      }
    }
    init()
  }, [])

  // Load embedding data
  useEffect(() => {
    if (!aresInitialized || !aresState?.has_embeddings) return

    const loadEmbeddings = async () => {
      setLoadingEmbeddings(true)
      try {
        const embeddingKeys = ['task_language_instruction', 'description_estimate']
        const embeddingDataMap: Record<string, any> = {}
        
        for (const key of embeddingKeys) {
          try {
            const response = await fetch(`/api/ares/embeddings/${key}`)
            if (response.ok) {
              const data = await response.json()
              embeddingDataMap[key] = data
              if (!activeEmbeddingTab) {
                setActiveEmbeddingTab(key)
              }
            }
          } catch (err) {
            console.error(`Error loading embedding ${key}:`, err)
          }
        }
        
        setEmbeddingData(embeddingDataMap)
      } catch (err) {
        console.error('Error loading embeddings:', err)
      } finally {
        setLoadingEmbeddings(false)
      }
    }

    loadEmbeddings()
  }, [aresInitialized, aresState?.has_embeddings])

  // Load distributions
  useEffect(() => {
    if (!aresInitialized) return

    const loadDistributions = async () => {
      try {
        const distResponse = await fetch('/api/ares/distributions')
        if (distResponse.ok) {
          const distData = await distResponse.json()
          const vizs = distData.visualizations || []
          setAresDistributions(vizs)
        }
      } catch (err: any) {
        console.error('Error loading distributions:', err)
      }
    }

    loadDistributions()
  }, [aresInitialized])

  return (
    <div className="min-h-screen bg-[#1a1a1a] text-[#d4d4d4]">
      {/* Header */}
      <header className="border-b border-[#2a2a2a] bg-[#222222] sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-between h-10">
            <div className="flex items-center gap-4">
              <h1 className="text-sm font-medium tracking-wide text-white">
                6d labs
              </h1>
              {currentDataset && (
                <>
                  <div className="h-3 w-px bg-[#2a2a2a]" />
                  <span className="text-xs text-[#8a8a8a] font-mono">
                    {currentDataset}
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Upload Data Section */}
        <div className="mb-8">
          <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Upload Data</h2>
          <DatasetUpload onDatasetLoaded={handleDatasetLoaded} />
        </div>

        {error && (
          <div className="mb-4 p-2.5 bg-[#2a1a1a] border border-[#3a2a2a] flex items-center gap-2">
            <XCircle className="w-3.5 h-3.5 text-[#cc6666]" />
            <span className="text-xs text-[#cc6666]">{error}</span>
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-5 h-5 animate-spin text-[#666666]" />
          </div>
        )}

        {/* Embedding Data Filters Section */}
        {aresInitialized && aresState?.has_embeddings && (
          <div className="mb-8">
            <div className="bg-[#222222] border border-[#2a2a2a] p-6">
              <h2 className="text-xs font-medium mb-4 text-[#d4d4d4]">Embedding Data Filters</h2>
              
              {Object.keys(embeddingData).length === 0 ? (
                <div className="text-xs text-[#8a8a8a] mb-4">
                  {loadingEmbeddings ? 'Loading embedding data...' : 'No embedding data available.'}
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
                            dragmode: 'lasso',
                            selectdirection: 'diagonal',
                          }}
                          config={{
                            displayModeBar: false,
                            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                          }}
                          style={{ width: '100%', height: '500px' }}
                          onSelected={(data: any) => {
                            if (data && data.points) {
                              const selectedIds = data.points
                                .map((p: any) => p.customdata?.[1] || p.customdata?.[0])
                                .filter((id: any) => id !== undefined)
                              
                              setEmbeddingSelections({
                                ...embeddingSelections,
                                [activeEmbeddingTab]: selectedIds,
                              })
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
          </div>
        )}

        {/* Data Analysis Section */}
        {currentDataset && !loading && (
          <div className="mb-8">
            <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Data Analysis</h2>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6 space-y-8">
              <DatasetOverview datasetInfo={datasetInfo} />
              <DatasetDistributions 
                datasetName={currentDataset} 
                aresDistributions={aresDistributions}
                aresInitialized={aresInitialized}
              />
              <EpisodePreview datasetData={datasetData} />
            </div>
          </div>
        )}

        {/* Dataset Curation Section */}
        {currentDataset && !loading && (
          <div className="mb-8">
            <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Dataset Curation</h2>
            <div className="bg-[#222222] border border-[#2a2a2a]">
              <div className="border-b border-[#2a2a2a]">
                <div className="flex">
                  <button
                    onClick={() => setCurationTab('automated')}
                    className={cn(
                      "px-4 py-2 text-sm font-medium transition-colors relative",
                      curationTab === 'automated'
                        ? "text-[#d4d4d4]"
                        : "text-[#8a8a8a] hover:text-[#b4b4b4]"
                    )}
                  >
                    Automated
                    {curationTab === 'automated' && (
                      <span className="absolute bottom-0 left-0 right-0 h-px bg-[#154e72]" />
                    )}
                  </button>
                  <button
                    onClick={() => setCurationTab('manual')}
                    className={cn(
                      "px-4 py-2 text-sm font-medium transition-colors relative",
                      curationTab === 'manual'
                        ? "text-[#d4d4d4]"
                        : "text-[#8a8a8a] hover:text-[#b4b4b4]"
                    )}
                  >
                    Manual
                    {curationTab === 'manual' && (
                      <span className="absolute bottom-0 left-0 right-0 h-px bg-[#154e72]" />
                    )}
                  </button>
                </div>
              </div>
              <div className="p-6">
                {curationTab === 'automated' ? (
                  <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4">
                    <AugmentationPanel
                      datasetName={currentDataset}
                      onComplete={handleAugmentationComplete}
                    />
                    <div className="mt-4 pt-4 border-t border-[#2a2a2a]">
                      <OptimizationPanel
                        datasetName={currentDataset}
                      />
                    </div>
                    <div className="mt-4 pt-4 border-t border-[#2a2a2a]">
                      <TestingPanel datasetName={currentDataset} />
                    </div>
                  </div>
                ) : (
                  <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4">
                    {/* Manual tab - empty for now */}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {!currentDataset && !loading && (
          <div className="text-center py-16">
            <p className="text-[#8a8a8a] text-xs">
              Load a dataset to get started
            </p>
          </div>
        )}
      </main>
    </div>
  )
}

