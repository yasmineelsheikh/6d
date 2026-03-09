'use client'

import React from 'react'

interface SettingsTabProps {
    environment: 'Indoor' | 'Outdoor' | ''
    isIndoor: boolean
    setIsIndoor: (v: boolean) => void
    isOutdoor: boolean
    setIsOutdoor: (v: boolean) => void
    selectedAxes: string[]
    setSelectedAxes: (v: string[]) => void
}

const INDOOR_AXES = ['Objects', 'Lighting', 'Color/Material']
const OUTDOOR_AXES = ['Objects', 'Lighting', 'Weather', 'Road Surface']

export default function SettingsTab({ environment, isIndoor, setIsIndoor, isOutdoor, setIsOutdoor, selectedAxes, setSelectedAxes }: SettingsTabProps) {
    const availableAxes = isIndoor ? INDOOR_AXES : isOutdoor ? OUTDOOR_AXES : []

    const toggleAxis = (axis: string) => {
        setSelectedAxes(
            selectedAxes.includes(axis)
                ? selectedAxes.filter(a => a !== axis)
                : [...selectedAxes, axis]
        )
    }

    return (
        <div className="max-w-xl space-y-8">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-white">Augmentation Settings</h3>

            <div className="space-y-6">
                {/* Environment */}
                <div>
                    <span className="text-[10px] uppercase tracking-widest text-[#d4d4d4] font-medium">Environment</span>
                    <div className="mt-3 flex items-center gap-4">
                        <label className="flex items-center gap-2 cursor-pointer group">
                            <input type="checkbox" checked={isIndoor} onChange={() => { setIsIndoor(!isIndoor); if (!isIndoor) setIsOutdoor(false) }}
                                className="w-4 h-4 rounded border-white/20 bg-white/5 text-[#4b6671] focus:ring-0 focus:ring-offset-0" />
                            <span className="text-sm text-white group-hover:text-white/80 transition-colors">Indoor</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer group">
                            <input type="checkbox" checked={isOutdoor} onChange={() => { setIsOutdoor(!isOutdoor); if (!isOutdoor) setIsIndoor(false) }}
                                className="w-4 h-4 rounded border-white/20 bg-white/5 text-[#4b6671] focus:ring-0 focus:ring-offset-0" />
                            <span className="text-sm text-white group-hover:text-white/80 transition-colors">Outdoor</span>
                        </label>
                    </div>
                </div>

                {/* Axes */}
                {availableAxes.length > 0 && (
                    <div>
                        <span className="text-[10px] uppercase tracking-widest text-[#d4d4d4] font-medium">Distribution Axes</span>
                        <div className="mt-3 flex flex-wrap gap-2">
                            {availableAxes.map(axis => (
                                <button key={axis} type="button" onClick={() => toggleAxis(axis)}
                                    className={`px-4 py-2 text-xs rounded-xl border transition-all ${selectedAxes.includes(axis)
                                        ? 'bg-white/10 border-white/30 text-white shadow-lg shadow-white/5'
                                        : 'bg-transparent border-white/10 text-white/60 hover:text-white hover:border-white/20'
                                        }`}>
                                    {axis}
                                </button>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
