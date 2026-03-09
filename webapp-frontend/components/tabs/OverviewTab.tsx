'use client'

import DatasetOverview from '@/components/DatasetOverview'
import DatasetDistributions from '@/components/DatasetDistributions'

interface OverviewTabProps {
    datasetName: string
    datasetInfo: any
    aresDistributions: any[]
    newDistributions: any[]
    distributionsLoading: boolean
    analysisView: 'original' | 'new'
    setAnalysisView: (v: 'original' | 'new') => void
    analysisSwitchEnabled: boolean
    variationsIncluded: boolean
}

export default function OverviewTab({
    datasetName, datasetInfo, aresDistributions, newDistributions,
    distributionsLoading, analysisView, setAnalysisView, analysisSwitchEnabled, variationsIncluded
}: OverviewTabProps) {
    return (
        <div className="space-y-6">
            {/* Original / New toggle */}
            <div className="flex items-center justify-end">
                <div className={`inline-flex rounded-lg border border-white/10 text-[11px] ${!analysisSwitchEnabled ? 'opacity-50 cursor-not-allowed' : ''}`}>
                    <button
                        type="button"
                        onClick={() => analysisSwitchEnabled && setAnalysisView('original')}
                        className={`px-3 py-1.5 rounded-l-lg transition-colors ${analysisView === 'original' ? 'bg-white/10 text-white' : 'text-[#9aa4b5]'}`}
                    >Original</button>
                    <button
                        type="button"
                        onClick={() => analysisSwitchEnabled && setAnalysisView('new')}
                        className={`px-3 py-1.5 rounded-r-lg border-l border-white/10 transition-colors ${analysisView === 'new' ? 'bg-white/10 text-white' : 'text-[#9aa4b5]'}`}
                    >New</button>
                </div>
            </div>

            <DatasetOverview datasetInfo={datasetInfo} />

            {analysisView === 'new' && !variationsIncluded && !distributionsLoading && (
                <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                    <p className="text-[#9aa4b5] text-xs">Run augmentation to see updated distributions.</p>
                </div>
            )}

            <DatasetDistributions
                datasetName={datasetName}
                aresDistributions={analysisView === 'new' ? newDistributions : aresDistributions}
                loading={distributionsLoading}
            />
        </div>
    )
}
