'use client'

interface DatasetInfo {
  dataset_name: string
  total_episodes: number
  robot_type: string
}

interface DatasetOverviewProps {
  datasetInfo: DatasetInfo | null
}

export default function DatasetOverview({ datasetInfo }: DatasetOverviewProps) {
  if (!datasetInfo) return null

  return (
    <div>
      <h2 className="text-xs font-medium mb-3 text-[#e3e8f0]">Dataset Overview</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-3">
          <div className="text-xs text-[#b5becb] mb-1">Total Episodes</div>
          <div className="text-lg font-medium text-[#e3e8f0]">
            {datasetInfo.total_episodes}
          </div>
        </div>
        <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-3">
          <div className="text-xs text-[#b5becb] mb-1">Robot Type</div>
          <div className="text-lg font-medium text-[#e3e8f0]">
            {datasetInfo.robot_type}
          </div>
        </div>
      </div>
    </div>
  )
}

