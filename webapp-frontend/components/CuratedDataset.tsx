'use client'

import { useState, useEffect } from 'react'
import { Download } from 'lucide-react'
import DatasetOverview from './DatasetOverview'
import DatasetDistributions from './DatasetDistributions'
import EpisodePreview from './EpisodePreview'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''

interface CuratedDatasetProps {
  datasetName: string
  datasetData: any[]
}

export default function CuratedDataset({ datasetName, datasetData }: CuratedDatasetProps) {
  const [datasetInfo, setDatasetInfo] = useState<any>(null)

  useEffect(() => {
    const loadInfo = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/datasets/${datasetName}/info`)
        if (response.ok) {
          const info = await response.json()
          setDatasetInfo(info)
        }
      } catch (error) {
        console.error('Failed to load dataset info:', error)
      }
    }
    loadInfo()
  }, [datasetName])

  const handleExport = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/datasets/${datasetName}/export?format=csv`)
      if (!response.ok) throw new Error('Export failed')
      const data = await response.json()
      
      // Create download link
      const blob = new Blob([data.content], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = data.filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error: any) {
      alert(error.message)
    }
  }

  if (datasetData.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-[#8a8a8a] text-xs">
          No curated dataset available. Run augmentation to generate curated data.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-medium text-[#d4d4d4]">Curated Dataset</h2>
        <button
          onClick={handleExport}
          className="px-3 py-1.5 text-xs text-white flex items-center gap-1.5 transition-colors"
          style={{ backgroundColor: '#4b6671' }}
        >
          <Download className="w-3.5 h-3.5" />
          Export Dataset
        </button>
      </div>
      <DatasetOverview datasetInfo={datasetInfo} />
      <DatasetDistributions datasetName={datasetName} isCurated={true} />
      <EpisodePreview datasetData={datasetData} />
    </div>
  )
}

