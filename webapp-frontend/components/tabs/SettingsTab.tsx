'use client'

import React from 'react'

interface SettingsTabProps {
    selectedAxes: string[]
    setSelectedAxes: (v: string[]) => void
    customAugmentationText: string
    setCustomAugmentationText: (v: string) => void
}

const ALL_AXES = ['Objects', 'Lighting', 'Color/Material', 'Weather', 'Road Surface']

export default function SettingsTab({ selectedAxes, setSelectedAxes, customAugmentationText, setCustomAugmentationText }: SettingsTabProps) {
    const toggleAxis = (axis: string) => {
        setSelectedAxes(
            selectedAxes.includes(axis)
                ? selectedAxes.filter(a => a !== axis)
                : [...selectedAxes, axis]
        )
    }

    const isCustomSelected = selectedAxes.includes('Custom')

    return (
        <div className="max-w-xl space-y-8">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-white">Augmentation Settings</h3>

            <div className="space-y-6">
                {/* Axes */}
                <div>
                    <span className="text-[10px] uppercase tracking-widest text-[#d4d4d4] font-medium">Axes</span>
                    <div className="mt-3 flex flex-wrap gap-2">
                        {ALL_AXES.map(axis => (
                            <button key={axis} type="button" onClick={() => toggleAxis(axis)}
                                className={`px-4 py-2 text-xs rounded-xl border transition-all ${selectedAxes.includes(axis)
                                    ? 'bg-white/10 border-white/30 text-white shadow-lg shadow-white/5'
                                    : 'bg-transparent border-white/10 text-white/60 hover:text-white hover:border-white/20'
                                    }`}>
                                {axis}
                            </button>
                        ))}
                        <button type="button" onClick={() => toggleAxis('Custom')}
                            className={`px-4 py-2 text-xs rounded-xl border transition-all ${isCustomSelected
                                ? 'bg-white/10 border-white/30 text-white shadow-lg shadow-white/5'
                                : 'bg-transparent border-white/10 text-white/60 hover:text-white hover:border-white/20'
                                }`}>
                            Custom
                        </button>
                    </div>
                    {isCustomSelected && (
                        <textarea
                            value={customAugmentationText}
                            onChange={e => setCustomAugmentationText(e.target.value)}
                            placeholder="Describe the augmentation you want..."
                            className="mt-3 w-full px-3 py-2.5 bg-white/5 border border-white/10 text-white text-xs placeholder:text-white/30 rounded-xl focus:outline-none focus:border-white/20 transition-colors resize-none"
                            rows={3}
                        />
                    )}
                </div>
            </div>
        </div>
    )
}
