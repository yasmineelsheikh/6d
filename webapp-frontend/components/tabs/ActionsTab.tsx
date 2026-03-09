'use client'

import { useState } from 'react'
import { ChevronRight, ChevronDown, Zap, ArrowUpDown, Upload, Search, BarChart3, Folder, Cloud } from 'lucide-react'
import AugmentationPanel from '@/components/AugmentationPanel'
import OptimizationPanel from '@/components/OptimizationPanel'
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
}

function AccordionPanel({ title, icon: Icon, statusText, children }: {
    title: string; icon: any; statusText: string; children: React.ReactNode
}) {
    const [isOpen, setIsOpen] = useState(false)
    return (
        <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
            <button
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center gap-3 px-5 py-4 text-left hover:bg-white/[0.03] transition-colors"
            >
                {isOpen ? <ChevronDown className="w-3.5 h-3.5 text-[#666] flex-shrink-0" /> : <ChevronRight className="w-3.5 h-3.5 text-[#666] flex-shrink-0" />}
                <Icon className="w-4 h-4 text-[#9aa4b5] flex-shrink-0" />
                <span className="text-xs font-medium text-white flex-1">{title}</span>
                <span className="text-[11px] text-[#555]">{statusText}</span>
            </button>
            {isOpen && (
                <div className="px-5 pb-5 pt-1 border-t border-white/5 transition-all">
                    {children}
                </div>
            )}
        </div>
    )
}

export default function ActionsTab(props: ActionsTabProps) {
    const [augMode, setAugMode] = useState<'automated' | 'manual'>('manual')

    return (
        <div className="space-y-3">
            {/* Impact Preview */}
            <div className="bg-[#4b6671]/10 border border-[#4b6671]/20 rounded-xl px-5 py-3 flex items-center gap-4">
                <BarChart3 className="w-4 h-4 text-[#4b6671] flex-shrink-0" />
                <span className="text-[11px] text-[#9aa4b5]">Select an action below to see projected dataset impact before confirming.</span>
            </div>

            {/* Augmentation */}
            <AccordionPanel title="Augmentation" icon={Zap} statusText="Ready">
                <div className="mb-3 flex items-center gap-2">
                    <span className="text-[10px] uppercase tracking-widest text-[#666] font-medium">Mode</span>
                    <div className="inline-flex rounded-lg border border-white/10 text-[11px]">
                        <button onClick={() => setAugMode('automated')} className={`px-3 py-1 rounded-l-lg transition-colors ${augMode === 'automated' ? 'bg-white/10 text-white' : 'text-[#9aa4b5]'}`}>Automated</button>
                        <button onClick={() => setAugMode('manual')} className={`px-3 py-1 rounded-r-lg border-l border-white/10 transition-colors ${augMode === 'manual' ? 'bg-white/10 text-white' : 'text-[#9aa4b5]'}`}>Manual</button>
                    </div>
                </div>
                <AugmentationPanel datasetName={props.datasetName} onComplete={props.onAugmentationComplete} />
            </AccordionPanel>

            {/* Optimisation */}
            <AccordionPanel title="Optimisation" icon={ArrowUpDown} statusText="Coming Soon">
                <div className="opacity-60 pointer-events-none">
                    <OptimizationPanel datasetName={props.datasetName} />
                </div>
            </AccordionPanel>

            {/* Add Data */}
            <AccordionPanel title="Add Data" icon={Upload} statusText="Ready">
                <div className="space-y-3">
                    <div className="flex items-center gap-2 border border-white/10 rounded-lg p-1 bg-[#1a1a1a] w-fit">
                        {(['local', 's3', 'huggingface'] as const).map((mode) => (
                            <button key={mode} type="button"
                                onClick={() => props.setUploadMode(mode)}
                                className={`px-3 py-1.5 text-xs flex items-center gap-1.5 rounded transition-colors ${props.uploadMode === mode ? 'bg-[#4b6671] text-white' : 'text-[#9aa4b5] hover:text-[#d4d4d4]'}`}
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
                                className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] text-xs cursor-pointer flex items-center gap-2 hover:bg-[#252525] transition-colors rounded-lg">
                                <Folder className="w-4 h-4 flex-shrink-0" />
                                <span className="flex-1 truncate">{props.datasetPath || 'Select folder...'}</span>
                            </label>
                        </div>
                    )}

                    {props.uploadMode === 's3' && (
                        <div className="flex flex-col gap-2">
                            <div className="grid grid-cols-2 gap-2">
                                <input type="text" placeholder="Access Key" value={props.s3AccessKey} onChange={e => props.setS3AccessKey(e.target.value)} className="px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666] text-xs rounded-lg focus:outline-none" />
                                <input type="password" placeholder="Secret Key" value={props.s3SecretKey} onChange={e => props.setS3SecretKey(e.target.value)} className="px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666] text-xs rounded-lg focus:outline-none" />
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                <input type="text" placeholder="Bucket" value={props.s3Bucket} onChange={e => props.setS3Bucket(e.target.value)} className="px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666] text-xs rounded-lg focus:outline-none" />
                                <input type="text" placeholder="Region" value={props.s3Region} onChange={e => props.setS3Region(e.target.value)} className="px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666] text-xs rounded-lg focus:outline-none" />
                            </div>
                            <input type="text" placeholder="Path within bucket" value={props.s3UserPath} onChange={e => { props.setS3UserPath(e.target.value); props.setDatasetPath(e.target.value) }} className="px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666] text-xs rounded-lg focus:outline-none" />
                        </div>
                    )}

                    {props.uploadMode === 'huggingface' && (
                        <div className="flex flex-col gap-2">
                            <input type="text" placeholder="Repository ID" value={props.hfRepoId} onChange={e => { props.setHfRepoId(e.target.value); props.setDatasetPath(e.target.value) }} className="px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666] text-xs rounded-lg focus:outline-none" />
                            <div className="grid grid-cols-2 gap-2">
                                <input type="text" placeholder="Split (default: train)" value={props.hfSplit} onChange={e => props.setHfSplit(e.target.value)} className="px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666] text-xs rounded-lg focus:outline-none" />
                                <input type="password" placeholder="HF Token (optional)" value={props.hfToken} onChange={e => props.setHfToken(e.target.value)} className="px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666] text-xs rounded-lg focus:outline-none" />
                            </div>
                        </div>
                    )}

                    <button onClick={props.handleLoadDataset} disabled={props.uploadLoading}
                        className="px-4 py-2 text-xs text-white bg-[#4b6671] hover:bg-[#3d5560] disabled:opacity-30 disabled:cursor-not-allowed transition-colors rounded-lg flex items-center gap-1.5">
                        {props.uploadLoading ? 'Loading...' : 'Upload'}
                    </button>
                </div>
            </AccordionPanel>

            {/* Scenario Explorer */}
            <AccordionPanel title="Scenario Explorer" icon={Search} statusText={`${props.datasetData.length} episodes`}>
                <EpisodePreview datasetData={props.datasetData} />
            </AccordionPanel>
        </div>
    )
}
