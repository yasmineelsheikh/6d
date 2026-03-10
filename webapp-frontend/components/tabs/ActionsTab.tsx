'use client'

import { useState } from 'react'
import { ChevronRight, ChevronDown, Folder, Cloud } from 'lucide-react'
import AugmentationPanel from '@/components/AugmentationPanel'
import OptimizationPanel from '@/components/OptimizationPanel'
import TestingPanel from '@/components/TestingPanel'
import EpisodePreview from '@/components/EpisodePreview'

interface ActionsTabProps {
    datasetName: string
    datasetData: any[]
    onAugmentationComplete: (name: string) => void
    // Upload state (shared with parent)
    uploadMode: 'local' | 's3' | 'huggingface'
    setUploadMode: (m: 'local' | 's3' | 'huggingface') => void
    uploadedFiles: FileList | null
    datasetPath: string
    handleFolderSelect: (e: React.ChangeEvent<HTMLInputElement>) => void
    s3AccessKey: string; setS3AccessKey: (v: string) => void
    s3SecretKey: string; setS3SecretKey: (v: string) => void
    s3Bucket: string; setS3Bucket: (v: string) => void
    s3Region: string; setS3Region: (v: string) => void
    s3UserPath: string; setS3UserPath: (v: string) => void
    hfRepoId: string; setHfRepoId: (v: string) => void
    hfSplit: string; setHfSplit: (v: string) => void
    hfToken: string; setHfToken: (v: string) => void
    handleLoadDataset: () => void
    uploadLoading: boolean
    uploadSuccess: boolean
    setDatasetPath: (v: string) => void
    setDatasetName: (v: string) => void
    // Augmentation settings (migrated from SettingsTab)
    isIndoor: boolean
    setIsIndoor: (v: boolean) => void
    isOutdoor: boolean
    setIsOutdoor: (v: boolean) => void
    selectedAxes: string[]
    setSelectedAxes: (v: string[]) => void
}

function AccordionPanel({ title, statusText, children }: {
    title: string; statusText: string; children: React.ReactNode
}) {
    const [isOpen, setIsOpen] = useState(false)
    return (
        <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
            <button
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center gap-3 px-5 py-4 text-left hover:bg-white/[0.03] transition-colors"
            >
                {isOpen ? <ChevronDown className="w-3.5 h-3.5 text-white/40 flex-shrink-0" /> : <ChevronRight className="w-3.5 h-3.5 text-white/40 flex-shrink-0" />}
                <span className="text-xs font-medium text-white flex-1">{title}</span>
                <span className="text-[11px] text-white/40">{statusText}</span>
            </button>
            {isOpen && (
                <div className="px-6 pb-6 pt-4 border-t border-white/5">
                    {children}
                </div>
            )}
        </div>
    )
}

const INDOOR_AXES = ['Objects', 'Lighting', 'Color/Material']
const OUTDOOR_AXES = ['Objects', 'Lighting', 'Weather', 'Road Surface']

const inputClass = "px-3 py-2.5 bg-[#1a1a1a] border border-white/10 text-white placeholder:text-white/30 text-xs rounded-xl focus:outline-none focus:border-white/20 transition-colors"

export default function ActionsTab(props: ActionsTabProps) {
    const availableAxes = props.isIndoor ? INDOOR_AXES : props.isOutdoor ? OUTDOOR_AXES : []

    const toggleAxis = (axis: string) => {
        props.setSelectedAxes(
            props.selectedAxes.includes(axis)
                ? props.selectedAxes.filter(a => a !== axis)
                : [...props.selectedAxes, axis]
        )
    }

    return (
        <div className="space-y-3">
            {/* Augmentation */}
            <AccordionPanel title="Augmentation" statusText="Ready">
                <AugmentationPanel datasetName={props.datasetName} onComplete={props.onAugmentationComplete} />

                {/* Augmentation Settings (migrated from Settings tab) */}
                <div className="mt-6 pt-5 border-t border-white/10 space-y-5">
                    <span className="text-[10px] uppercase tracking-widest text-white/40 font-medium">Augmentation Settings</span>

                    {/* Environment */}
                    <div>
                        <span className="text-[10px] uppercase tracking-widest text-white/30 font-medium">Environment</span>
                        <div className="mt-3 flex items-center gap-4">
                            <label className="flex items-center gap-2 cursor-pointer group">
                                <input type="checkbox" checked={props.isIndoor} onChange={() => { props.setIsIndoor(!props.isIndoor); if (!props.isIndoor) props.setIsOutdoor(false) }}
                                    className="w-4 h-4 rounded border-white/20 bg-white/5 text-[#4b6671] focus:ring-0 focus:ring-offset-0" />
                                <span className="text-sm text-white group-hover:text-white/80 transition-colors">Indoor</span>
                            </label>
                            <label className="flex items-center gap-2 cursor-pointer group">
                                <input type="checkbox" checked={props.isOutdoor} onChange={() => { props.setIsOutdoor(!props.isOutdoor); if (!props.isOutdoor) props.setIsIndoor(false) }}
                                    className="w-4 h-4 rounded border-white/20 bg-white/5 text-[#4b6671] focus:ring-0 focus:ring-offset-0" />
                                <span className="text-sm text-white group-hover:text-white/80 transition-colors">Outdoor</span>
                            </label>
                        </div>
                    </div>

                    {/* Distribution Axes */}
                    {availableAxes.length > 0 && (
                        <div>
                            <span className="text-[10px] uppercase tracking-widest text-white/30 font-medium">Distribution Axes</span>
                            <div className="mt-3 flex flex-wrap gap-2">
                                {availableAxes.map(axis => (
                                    <button key={axis} type="button" onClick={() => toggleAxis(axis)}
                                        className={`px-4 py-2 text-xs rounded-xl border transition-all ${props.selectedAxes.includes(axis)
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
            </AccordionPanel>

            {/* Optimisation */}
            <AccordionPanel title="Optimisation" statusText="Coming Soon">
                <div className="opacity-60 pointer-events-none">
                    <OptimizationPanel datasetName={props.datasetName} />
                </div>
            </AccordionPanel>

            {/* Add Data */}
            <AccordionPanel title="Add Data" statusText="Ready">
                <div className="space-y-4">
                    <div className="flex items-center gap-1.5 border border-white/10 rounded-xl p-1 bg-[#1a1a1a] w-fit">
                        {(['local', 's3', 'huggingface'] as const).map((mode) => (
                            <button key={mode} type="button"
                                onClick={() => props.setUploadMode(mode)}
                                className={`px-4 py-2 text-xs flex items-center gap-1.5 rounded-lg transition-colors ${props.uploadMode === mode ? 'bg-white/10 text-white' : 'text-white/60 hover:text-white'}`}
                            >
                                {mode === 'local' ? <Folder className="w-3 h-3" /> : <Cloud className="w-3 h-3" />}
                                {mode === 'local' ? 'Local' : mode === 's3' ? 'S3' : 'HF'}
                            </button>
                        ))}
                    </div>

                    {props.uploadMode === 'local' && (
                        <div className="relative">
                            <input type="file" id="folder-upload-actions"
                                {...({ webkitdirectory: '', directory: '' } as React.InputHTMLAttributes<HTMLInputElement>)}
                                multiple onChange={props.handleFolderSelect} className="hidden" />
                            <label htmlFor="folder-upload-actions"
                                className={`w-full ${inputClass} cursor-pointer flex items-center gap-2 hover:bg-white/[0.03]`}>
                                <Folder className="w-4 h-4 flex-shrink-0 text-white/40" />
                                <span className="flex-1 truncate">{props.datasetPath || 'Select folder...'}</span>
                            </label>
                        </div>
                    )}

                    {props.uploadMode === 's3' && (
                        <div className="flex flex-col gap-3">
                            <div className="grid grid-cols-2 gap-3">
                                <input type="text" placeholder="Access Key" value={props.s3AccessKey} onChange={e => props.setS3AccessKey(e.target.value)} className={inputClass} />
                                <input type="password" placeholder="Secret Key" value={props.s3SecretKey} onChange={e => props.setS3SecretKey(e.target.value)} className={inputClass} />
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                <input type="text" placeholder="Bucket" value={props.s3Bucket} onChange={e => props.setS3Bucket(e.target.value)} className={inputClass} />
                                <input type="text" placeholder="Region" value={props.s3Region} onChange={e => props.setS3Region(e.target.value)} className={inputClass} />
                            </div>
                            <input type="text" placeholder="Path within bucket" value={props.s3UserPath} onChange={e => { props.setS3UserPath(e.target.value); props.setDatasetPath(e.target.value) }} className={inputClass} />
                        </div>
                    )}

                    {props.uploadMode === 'huggingface' && (
                        <div className="flex flex-col gap-3">
                            <input type="text" placeholder="Repository ID (e.g. org/dataset)" value={props.hfRepoId} onChange={e => { props.setHfRepoId(e.target.value); props.setDatasetPath(e.target.value) }} className={inputClass} />
                            <div className="grid grid-cols-2 gap-3">
                                <input type="text" placeholder="Split (default: train)" value={props.hfSplit} onChange={e => props.setHfSplit(e.target.value)} className={inputClass} />
                                <input type="password" placeholder="HF Token (optional)" value={props.hfToken} onChange={e => props.setHfToken(e.target.value)} className={inputClass} />
                            </div>
                        </div>
                    )}

                    <button onClick={props.handleLoadDataset} disabled={props.uploadLoading}
                        className="px-5 py-2.5 text-xs text-white bg-gradient-to-r from-[#4b6671] to-[#3d5f6f] hover:from-[#567a86] hover:to-[#4b6f7f] disabled:opacity-30 disabled:cursor-not-allowed transition-all rounded-xl flex items-center gap-1.5">
                        {props.uploadLoading ? 'Uploading...' : 'Upload'}
                    </button>
                </div>
            </AccordionPanel>

            {/* Import Test Runs */}
            <AccordionPanel title="Import Test Runs" statusText="Ready">
                <TestingPanel datasetName={props.datasetName} />
            </AccordionPanel>

            {/* Data Explorer */}
            <AccordionPanel title="Data Explorer" statusText={`${props.datasetData.length} episodes`}>
                <EpisodePreview datasetData={props.datasetData} />
            </AccordionPanel>
        </div>
    )
}
