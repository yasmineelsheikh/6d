'use client'

import { Upload, Folder, Cloud } from 'lucide-react'
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
    hasNoData?: boolean
    // Upload props for fallback
    uploadMode?: 'local' | 's3' | 'huggingface'
    setUploadMode?: (m: 'local' | 's3' | 'huggingface') => void
    uploadedFiles?: FileList | null
    datasetPath?: string
    handleFolderSelect?: (e: React.ChangeEvent<HTMLInputElement>) => void
    s3AccessKey?: string; setS3AccessKey?: (v: string) => void
    s3SecretKey?: string; setS3SecretKey?: (v: string) => void
    s3Bucket?: string; setS3Bucket?: (v: string) => void
    s3Region?: string; setS3Region?: (v: string) => void
    s3UserPath?: string; setS3UserPath?: (v: string) => void
    hfRepoId?: string; setHfRepoId?: (v: string) => void
    hfSplit?: string; setHfSplit?: (v: string) => void
    hfToken?: string; setHfToken?: (v: string) => void
    handleLoadDataset?: () => void
    uploadLoading?: boolean
    uploadSuccess?: boolean
    setDatasetPath?: (v: string) => void
    setDatasetName?: (v: string) => void
}

export default function OverviewTab(props: OverviewTabProps) {
    const {
        datasetName, datasetInfo, aresDistributions, newDistributions,
        distributionsLoading, analysisView, setAnalysisView, analysisSwitchEnabled, variationsIncluded,
        hasNoData
    } = props

    // No data state — show upload prompt
    if (hasNoData) {
        const mode = props.uploadMode || 'local'
        return (
            <div className="space-y-6">
                <div className="flex flex-col items-center justify-center py-16">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-white/[0.06] to-white/[0.02] border border-white/10 flex items-center justify-center mb-5">
                        <Upload className="w-7 h-7 text-white/60" />
                    </div>
                    <h3 className="text-base font-medium text-white mb-1.5">No dataset uploaded yet</h3>
                    <p className="text-[13px] text-white/50 mb-8 text-center max-w-sm">Upload your data to start seeing distributions, episode previews, and run augmentations.</p>

                    {/* Upload form */}
                    <div className="w-full max-w-lg space-y-4">
                        {/* Mode toggle */}
                        <div className="flex items-center justify-center gap-1 border border-white/10 rounded-xl p-1 bg-[#1a1a1a] w-fit mx-auto">
                            {(['local', 's3', 'huggingface'] as const).map((m) => (
                                <button key={m} type="button"
                                    onClick={() => props.setUploadMode?.(m)}
                                    className={`px-4 py-2 text-xs flex items-center gap-1.5 rounded-lg transition-colors ${mode === m ? 'bg-white/10 text-white' : 'text-white/40 hover:text-white'}`}
                                >
                                    {m === 'local' ? <Folder className="w-3 h-3" /> : <Cloud className="w-3 h-3" />}
                                    {m === 'local' ? 'Local' : m === 's3' ? 'S3' : 'Hugging Face'}
                                </button>
                            ))}
                        </div>

                        {/* Local */}
                        {mode === 'local' && (
                            <div className="relative">
                                <input type="file" id="folder-upload-overview"
                                    {...({ webkitdirectory: '', directory: '' } as React.InputHTMLAttributes<HTMLInputElement>)}
                                    multiple onChange={props.handleFolderSelect} className="hidden" />
                                <label htmlFor="folder-upload-overview"
                                    className="w-full px-4 py-3 bg-[#1a1a1a] border border-dashed border-white/15 text-white text-xs cursor-pointer flex items-center gap-3 hover:bg-white/[0.03] hover:border-white/20 transition-all rounded-xl">
                                    <Folder className="w-4 h-4 flex-shrink-0 text-white/40" />
                                    <span className="flex-1 truncate">{props.datasetPath || 'Select folder to upload...'}</span>
                                </label>
                            </div>
                        )}

                        {/* S3 */}
                        {mode === 's3' && (
                            <div className="flex flex-col gap-2">
                                <div className="grid grid-cols-2 gap-2">
                                    <input type="text" placeholder="Access Key" value={props.s3AccessKey || ''} onChange={e => props.setS3AccessKey?.(e.target.value)} className="px-3 py-2.5 bg-[#1a1a1a] border border-white/10 text-white placeholder:text-white/20 text-xs rounded-xl focus:outline-none focus:border-white/20" />
                                    <input type="password" placeholder="Secret Key" value={props.s3SecretKey || ''} onChange={e => props.setS3SecretKey?.(e.target.value)} className="px-3 py-2.5 bg-[#1a1a1a] border border-white/10 text-white placeholder:text-white/20 text-xs rounded-xl focus:outline-none focus:border-white/20" />
                                </div>
                                <div className="grid grid-cols-2 gap-2">
                                    <input type="text" placeholder="Bucket" value={props.s3Bucket || ''} onChange={e => props.setS3Bucket?.(e.target.value)} className="px-3 py-2.5 bg-[#1a1a1a] border border-white/10 text-white placeholder:text-white/20 text-xs rounded-xl focus:outline-none focus:border-white/20" />
                                    <input type="text" placeholder="Region" value={props.s3Region || ''} onChange={e => props.setS3Region?.(e.target.value)} className="px-3 py-2.5 bg-[#1a1a1a] border border-white/10 text-white placeholder:text-white/20 text-xs rounded-xl focus:outline-none focus:border-white/20" />
                                </div>
                                <input type="text" placeholder="Path within bucket" value={props.s3UserPath || ''} onChange={e => { props.setS3UserPath?.(e.target.value); props.setDatasetPath?.(e.target.value) }} className="px-3 py-2.5 bg-[#1a1a1a] border border-white/10 text-white placeholder:text-white/20 text-xs rounded-xl focus:outline-none focus:border-white/20" />
                            </div>
                        )}

                        {/* Hugging Face */}
                        {mode === 'huggingface' && (
                            <div className="flex flex-col gap-2">
                                <input type="text" placeholder="Repository ID (e.g. org/dataset)" value={props.hfRepoId || ''} onChange={e => { props.setHfRepoId?.(e.target.value); props.setDatasetPath?.(e.target.value) }} className="px-3 py-2.5 bg-[#1a1a1a] border border-white/10 text-white placeholder:text-white/20 text-xs rounded-xl focus:outline-none focus:border-white/20" />
                                <div className="grid grid-cols-2 gap-2">
                                    <input type="text" placeholder="Split (default: train)" value={props.hfSplit || ''} onChange={e => props.setHfSplit?.(e.target.value)} className="px-3 py-2.5 bg-[#1a1a1a] border border-white/10 text-white placeholder:text-white/20 text-xs rounded-xl focus:outline-none focus:border-white/20" />
                                    <input type="password" placeholder="HF Token (optional)" value={props.hfToken || ''} onChange={e => props.setHfToken?.(e.target.value)} className="px-3 py-2.5 bg-[#1a1a1a] border border-white/10 text-white placeholder:text-white/20 text-xs rounded-xl focus:outline-none focus:border-white/20" />
                                </div>
                            </div>
                        )}

                        <button onClick={props.handleLoadDataset} disabled={props.uploadLoading}
                            className="w-full py-3 text-xs font-medium text-white bg-gradient-to-r from-[#4b6671] to-[#3d5f6f] hover:from-[#567a86] hover:to-[#4b6f7f] disabled:opacity-30 disabled:cursor-not-allowed transition-all rounded-xl flex items-center justify-center gap-2 shadow-lg shadow-[#4b6671]/10">
                            <Upload className="w-3.5 h-3.5" />
                            {props.uploadLoading ? 'Uploading...' : 'Upload Dataset'}
                        </button>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Original / New toggle */}
            <div className="flex items-center justify-end">
                <div className={`inline-flex rounded-lg border border-white/10 text-[11px] ${!analysisSwitchEnabled ? 'opacity-50 cursor-not-allowed' : ''}`}>
                    <button
                        type="button"
                        onClick={() => analysisSwitchEnabled && setAnalysisView('original')}
                        className={`px-3 py-1.5 rounded-l-lg transition-colors ${analysisView === 'original' ? 'bg-white/10 text-white' : 'text-white/40'}`}
                    >Original</button>
                    <button
                        type="button"
                        onClick={() => analysisSwitchEnabled && setAnalysisView('new')}
                        className={`px-3 py-1.5 rounded-r-lg border-l border-white/10 transition-colors ${analysisView === 'new' ? 'bg-white/10 text-white' : 'text-white/40'}`}
                    >New</button>
                </div>
            </div>

            <DatasetOverview datasetInfo={datasetInfo} />

            {analysisView === 'new' && !variationsIncluded && !distributionsLoading && (
                <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                    <p className="text-white text-xs">Run augmentation to see updated distributions.</p>
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
