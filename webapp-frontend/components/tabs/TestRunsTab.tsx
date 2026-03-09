'use client'

import TestingPanel from '@/components/TestingPanel'
import EpisodePreview from '@/components/EpisodePreview'

interface TestRunsTabProps {
    datasetName: string
    datasetData: any[]
}

export default function TestRunsTab({ datasetName, datasetData }: TestRunsTabProps) {
    return (
        <div className="space-y-6">
            <div>
                <span className="text-[10px] uppercase tracking-widest text-white/30 font-medium">Upload Test Data</span>
                <div className="mt-3 bg-white/5 border border-white/10 rounded-xl p-5">
                    <TestingPanel datasetName={datasetName} />
                </div>
            </div>

            <div>
                <span className="text-[10px] uppercase tracking-widest text-white/30 font-medium">Episode Browser</span>
                <div className="mt-3 bg-white/5 border border-white/10 rounded-xl p-5">
                    <EpisodePreview datasetData={datasetData} />
                </div>
            </div>
        </div>
    )
}
