'use client'

import { useState } from 'react'
import { Play, Loader2, CheckCircle2 } from 'lucide-react'

const API_BASE = 'https://6d-nu.vercel.app'

interface OptimizationPanelProps {
  datasetName: string
  onComplete?: (datasetName: string) => void
}

export default function OptimizationPanel({ datasetName, onComplete }: OptimizationPanelProps) {
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [result, setResult] = useState<string | null>(null)

  const handleRun = async () => {
    setLoading(true)
    setSuccess(false)
    setResult(null)

    try {
      const response = await fetch(`${API_BASE}/api/optimization/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: datasetName,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Optimization failed')
      }

      const data = await response.json()
      setResult(data.result || 'Optimization completed')
      setSuccess(true)
      if (onComplete) {
        onComplete(datasetName)
      }
    } catch (error: any) {
      alert(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <button
          onClick={handleRun}
          disabled={loading}
          className="w-full px-3 py-2 text-xs text-white disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center gap-1.5 transition-colors"
          style={{ backgroundColor: '#4b6671' }}
        >
          {loading ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Running optimization...
            </>
          ) : success ? (
            <>
              <CheckCircle2 className="w-3.5 h-3.5" />
              Complete
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5" />
              Execute
            </>
          )}
        </button>
        {result && (
          <div className="mt-3 p-2.5 bg-[#1a2a1a] border border-[#2a3a2a]">
            <p className="text-xs text-[#88aa88] mb-1">
              Optimization result:
            </p>
            <p className="text-xs text-[#88aa88] font-mono break-all">{result}</p>
          </div>
        )}
    </div>
  )
}

