'use client'

import { Clock, Upload, Zap, TrendingUp, ArrowUpDown } from 'lucide-react'

interface HistoryEntry {
    id: string
    type: 'upload' | 'augmentation' | 'optimisation' | 'test_import'
    description: string
    timestamp: string
    samplesChanged?: string
    performanceDelta?: string
}

const SEED_HISTORY: HistoryEntry[] = [
    { id: 'h1', type: 'upload', description: 'Initial dataset uploaded', timestamp: '2026-01-15', samplesChanged: '+80 samples' },
    { id: 'h2', type: 'augmentation', description: 'Augmentation run — lighting variations', timestamp: '2026-01-22', samplesChanged: '+240 samples', performanceDelta: '+3%' },
    { id: 'h3', type: 'test_import', description: 'Test run data imported', timestamp: '2026-02-01', samplesChanged: '+40 samples' },
    { id: 'h4', type: 'upload', description: 'Additional data uploaded (v2)', timestamp: '2026-02-03', samplesChanged: '+120 samples', performanceDelta: '-1%' },
    { id: 'h5', type: 'optimisation', description: 'Redundancy optimisation pass', timestamp: '2026-02-08', samplesChanged: '-35 samples', performanceDelta: '+0.5%' },
]

const TYPE_ICONS = {
    upload: Upload,
    augmentation: Zap,
    optimisation: ArrowUpDown,
    test_import: TrendingUp,
}

const TYPE_COLORS = {
    upload: 'text-[#5fa35f]',
    augmentation: 'text-[#c0a854]',
    optimisation: 'text-[#4b6671]',
    test_import: 'text-[#9a7bd4]',
}

export default function HistoryTab() {
    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-widest text-white/30 font-medium">Dataset Timeline</span>
                <button className="text-[11px] text-white/30 hover:text-white/50 transition-colors cursor-not-allowed" disabled>
                    Compare Versions
                </button>
            </div>

            <div className="relative">
                {/* Timeline line */}
                <div className="absolute left-[15px] top-0 bottom-0 w-px bg-white/10" />

                <div className="space-y-3">
                    {SEED_HISTORY.map((entry) => {
                        const Icon = TYPE_ICONS[entry.type]
                        const iconColor = TYPE_COLORS[entry.type]
                        return (
                            <div key={entry.id} className="relative flex items-start gap-4 pl-0">
                                <div className={`relative z-10 w-8 h-8 rounded-full bg-white/5 border border-white/10 flex items-center justify-center flex-shrink-0 ${iconColor}`}>
                                    <Icon className="w-3.5 h-3.5" />
                                </div>
                                <div className="flex-1 bg-white/5 border border-white/10 rounded-xl p-4">
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-xs text-white">{entry.description}</span>
                                        <span className="text-[11px] text-white/30 font-mono">{entry.timestamp}</span>
                                    </div>
                                    <div className="flex items-center gap-3 text-[11px] text-white/40">
                                        {entry.samplesChanged && <span>{entry.samplesChanged}</span>}
                                        {entry.performanceDelta && (
                                            <span className={entry.performanceDelta.startsWith('+') ? 'text-[#5fa35f]' : 'text-[#cc6666]'}>
                                                {entry.performanceDelta} success rate
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>
        </div>
    )
}
